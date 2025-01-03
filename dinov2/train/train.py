import argparse
import logging
import math
import os
from functools import partial

import torch
import torch.nn.functional as F

from fvcore.common.checkpoint import PeriodicCheckpointer

# ======== These are typical DINOv2 imports (paths may vary in your repo) ========
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

# ======== Import your new meta-arch with text encoder & contrastive ========
from ssl_meta_arch import SSLMetaArch  # <--- the file you updated to include text_encoder & do_contrastive


torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training with text", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--resume", action="store_true", help="Attempt to resume from checkpoint directory.")
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--output-dir", default="", type=str, help="Output directory for logs & checkpoints")
    parser.add_argument("--img_format", type=str, default=".png")
    parser.add_argument("--max_to_keep", type=int, default=3)
    parser.add_argument("--save_frequency", type=int, default=3)
    return parser


def build_optimizer(cfg, params_groups):
    # same as standard DINO
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    # same as standard DINO
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    # freeze last layer
    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = 0

    logger.info("Schedulers ready.")
    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    # Placeholder for any evaluation
    new_state_dict = model.teacher.state_dict()
    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False, max_to_keep=3, save_frequency=3):
    """
    Updated training loop that also reads text tokens from your dataset
    and calls model.forward_backward(...) with text tokens.
    """
    model.train()
    inputs_dtype = torch.half if cfg.train.get("use_fp16", True) else torch.float
    fp16_scaler = model.fp16_scaler

    # Setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # Checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    if resume:
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    else:
        start_iter = 0

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=save_frequency * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=max_to_keep,
    )

    # Setup data augmentation & collate
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * n_tokens,
    )
    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # Build dataset that yields (image, text) pairs
    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,  # must produce (image, text)
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # Sampler
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    iteration = start_iter
    logger.info(f"Starting training from iteration {start_iter}")
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data_batch in metric_logger.log_every(data_loader, 10, header, max_iter, start_iter):
        current_batch_size = data_batch["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # zero grad
        optimizer.zero_grad(set_to_none=True)

        # images
        data_batch["collated_global_crops"] = data_batch["collated_global_crops"].cuda(non_blocking=True)
        # local crops optional

        # text tokens - must be included in your dataset or collate
        text_tokens = data_batch.get("text_tokens", None)  # ensure your dataset provides this
        if text_tokens is not None:
            text_tokens = text_tokens.cuda(non_blocking=True)

        # forward & backward
        loss_dict = model.forward_backward(data_batch, teacher_temp=teacher_temp, text_tokens=text_tokens)

        # clip / scale
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # update teacher
        model.update_teacher(mom)

        # gather losses for logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)

        # Convert them to float
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}
        # sum all losses
        losses_reduced = sum(loss_dict_reduced.values())

        metric_logger.update(
            lr=lr,
            wd=wd,
            mom=mom,
            last_layer_lr=last_layer_lr,
            current_batch_size=current_batch_size,
            total_loss=losses_reduced,
            **loss_dict_reduced,
        )

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError

        # test / checkpoint
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration)
        iteration += 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    # Build model from your new meta-arch
    # This model has text_encoder integrated, do_contrastive = True, etc.
    model = SSLMetaArch(cfg).cuda()
    model.prepare_for_distributed_training()

    if args.eval_only:
        iteration = FSDPCheckpointer(model, save_dir=cfg.train.output_dir)\
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)\
            .get("iteration", -1) + 1
        do_test(cfg, model, f"manual_{iteration}")
        return

    do_train(
        cfg,
        model,
        resume=args.resume,
        max_to_keep=args.max_to_keep,
        save_frequency=args.save_frequency
    )


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
