# dino_text_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional

###################################################
# Example CLIP-like Text Transformer in DINO style
###################################################
class DinoTextEncoder(nn.Module):
    """
    A minimal text encoder, adapted from CLIP's text transformer.
    You might copy or simplify code from your CLIP model's text transformer.
    """

    def __init__(self,
                 vocab_size: int = 49408,
                 max_len: int = 77,
                 embed_dim: int = 512,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 pad_id: int = 0):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pad_id = pad_id

        # Basic token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # A small stack of Transformer blocks (attention + MLP)
        self.blocks = nn.ModuleList([
            DinoTextBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)

        # optional projection head
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        text_tokens: (B, L) integer IDs
        returns: text_emb of shape (B, embed_dim)
        """
        # 1) Embed tokens + positions
        x = self.token_embedding(text_tokens)  # (B, L, D)
        B, L, D = x.shape
        # add learned positional embedding
        # (clamp L at max_len in case we have short or longer sequences)
        pos_emb = self.positional_embedding[:, :L, :]
        x = x + pos_emb

        # 2) Pass through blocks
        for blk in self.blocks:
            x = blk(x)

        # 3) Final layer norm
        x = self.ln_final(x)

        # 4) Pool the last token or [CLS]-like approach. Simplify by taking x[:, 0, :] or x.mean(dim=1)
        # For CLIP, we often take x[:, -1, :] as the [EOS] pooling. Let's do that for demonstration:
        text_emb = x[:, -1, :]  # shape (B, D)

        # 5) Optional projection
        text_emb = self.proj(text_emb)  # (B, D)

        return text_emb


class DinoTextBlock(nn.Module):
    """
    Minimal text block: self-attention + MLP
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        # self-attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x
