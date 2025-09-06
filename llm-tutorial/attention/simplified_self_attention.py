import torch

from base_attention import BaseAttention


class SimplifiedSelfAttention(BaseAttention):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        batch_size, cxt_len, d_out = x.shape
        attn_scores = x @ x.transpose(1, 2)  # 0: batch_size, 1: cxt_len, 2: d_out
        attn_weights = torch.softmax(attn_scores, dim=-1)
        all_context_vecs = attn_weights @ x
        return all_context_vecs
