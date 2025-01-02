# ssl_meta_arch.py
#
# This file is a modified version of the original "dinov2.ssl_meta_arch" that adds:
# 1) A DinoTextEncoder instance (self.text_encoder).
# 2) A forward pass for text inside the forward_backward() method, which uses the new text encoder.
# 3) Logic to handle a contrastive loss with the text embeddings (you'll integrate that separately in the loss).

import copy
import logging
import math

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model
from dinov2.models.vision_transformer import BlockChunk

from dino_text_encoder import DinoTextEncoder  # <-- (1) we import our custom text encoder

logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = {}
        teacher_model_dict = {}

        # -----------------------------------------------------------
        # Build the backbone (DINO) for student & teacher
        # -----------------------------------------------------------
        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head_fn = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                from dinov2.loss import KoLeoLoss
                self.koleo_loss = KoLeoLoss()
        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            # add normal dino heads
            student_model_dict["dino_head"] = dino_head_fn()
            teacher_model_dict["dino_head"] = dino_head_fn()

        # -----------------------------------------------------------
        # iBOT logic removed for brevity... keep if needed
        # -----------------------------------------------------------

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # freeze teacher params
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: both are {cfg.student.arch} network.")

        # -----------------------------------------------------------
        # (2) Add Text Encoder from dino_text_encoder
        # -----------------------------------------------------------
        text_embed_dim = cfg.get("text_embed_dim", 512)
        text_vocab_size = cfg.get("text_vocab_size", 49408)
        text_num_layers = cfg.get("text_num_layers", 12)
        text_num_heads = cfg.get("text_num_heads", 8)
        text_mlp_ratio = cfg.get("text_mlp_ratio", 4.0)
        text_max_len = cfg.get("text_max_len", 77)

        self.text_encoder = DinoTextEncoder(
            vocab_size=text_vocab_size,
            max_len=text_max_len,
            embed_dim=text_embed_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            mlp_ratio=text_mlp_ratio
        )
        # note: no teacher text encoder by default

        # store a flag if we want to do the contrastive objective
        self.do_contrastive = cfg.get("contrastive_loss_weight", 0.0) > 0

        self.need_to_synchronize_fsdp_streams = True

    def forward(self, inputs):
        """
        Typically not used in DINO; see forward_backward for the main logic.
        But let's keep the signature for compatibility.
        """
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    # ------------------------------------------------------
    # (3) The main forward_backward that now also handles text
    # ------------------------------------------------------
    def forward_backward(self, images, teacher_temp, text_tokens=None):
        """
        We expect either:
          - images: dict with global/local crops
          - text_tokens: Optional[torch.Tensor] for the text encoder
        Then we compute the normal DINO losses, plus optionally a contrastive loss with text.
        """

        # original code for image crops, teacher forward, etc....
        # [redacted for brevity, see original code]
        # We'll do a simplified version:
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        # local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        # etc...

        # teacher backbone
        with torch.no_grad():
            teacher_out = self.teacher["backbone"](global_crops, is_training=True)
        teacher_cls_tokens = teacher_out["x_norm_clstoken"]

        # student backbone
        student_out = self.student["backbone"](global_crops, is_training=True)
        student_cls_tokens = student_out["x_norm_clstoken"]

        # pass student_cls_tokens into dino_head
        student_cls_after_head = self.student["dino_head"](student_cls_tokens)
        teacher_cls_after_head = self.teacher["dino_head"](teacher_cls_tokens)

        # DINO teacher softmax
        teacher_softmaxed_centered = self.dino_loss.softmax_center_teacher(
            teacher_cls_after_head, teacher_temp=teacher_temp
        )
        self.dino_loss.update_center(teacher_cls_after_head)

        # dino cross-entropy
        dino_loss = self.dino_loss([student_cls_after_head], [teacher_softmaxed_centered])

        total_loss = dino_loss
        loss_dict = {"dino_loss": dino_loss.detach()}

        # (4) If we also want a text contrastive objective:
        if self.do_contrastive and text_tokens is not None:
            text_emb = self.text_encoder(text_tokens)  # shape (B, text_embed_dim)
            # you can compute e.g. clip-style cross-entropy:
            contrastive_loss_val = self.compute_contrastive_loss(student_cls_tokens, text_emb)
            total_loss = total_loss + self.cfg.get("contrastive_loss_weight", 0.5) * contrastive_loss_val
            loss_dict["contrastive_loss"] = contrastive_loss_val.detach()

        # final backprop
        self.backprop_loss(total_loss)

        # you can do teacher update, or rely on update_teacher() method
        # self.update_teacher(m=0.99) for example

        return loss_dict

    def compute_contrastive_loss(self, image_emb, text_emb):
        """
        A minimal CLIP-like contrastive. If you'd prefer a separate loss class, do that.
        """
        image_emb_norm = nn.functional.normalize(image_emb, dim=-1)
        text_emb_norm = nn.functional.normalize(text_emb, dim=-1)

        # temperature from config or a param
        clip_temp = self.cfg.get("clip_temperature", 0.07)
        logits = (image_emb_norm @ text_emb_norm.t()) / clip_temp
        B = image_emb.size(0)
        labels = torch.arange(B, device=image_emb.device)
        loss_i2t = nn.functional.cross_entropy(logits, labels)
        loss_t2i = nn.functional.cross_entropy(logits.t(), labels)
        contrastive_loss = 0.5 * (loss_i2t + loss_t2i)
        return contrastive_loss

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        # same as original code, update teacher from student
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        # for param groups with decay, etc.
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        # gather param groups from student submodules
        # plus the text encoder
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        # we also do text_encoder if you want separate param groups
        if hasattr(self, "text_encoder"):
            # optional: or just treat it as part of the same param group
            text_groups = get_params_groups_with_decay(self.text_encoder)
            all_params_groups += fuse_params_groups(text_groups)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])

        # if you want the text encoder under FSDP, do something similar:
        # text_model_cfg = self.cfg.compute_precision.student.get("text_encoder", None)
        # if text_model_cfg:
        #     self.text_encoder = get_fsdp_wrapper(text_model_cfg, ...)(self.text_encoder)
