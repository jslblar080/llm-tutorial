import torch
import torch.nn as nn

from ..base_attention import BaseAttention


class SelfAttention(BaseAttention):

    _W_query: nn.Linear
    _W_key: nn.Linear
    _W_value: nn.Linear

    def __init__(self, seed_num: int, d_in: int, d_out: int, qkv_bias=False):
        super().__init__()
        torch.manual_seed(seed_num)
        # nn.Linear use optimized weight initialization scheme
        # nn.Linear effectively performs matrix multiplication when the bias units are disabled
        self._W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        batch_size, cxt_len, d_in = x.shape
        queries = self._W_query(x)
        keys = self._W_key(x)
        values = self._W_value(x)
        attn_scores = queries @ keys.transpose(
            1, 2  # 0: batch_size, 1: cxt_len, 2: d_out
        )
        # normalize by embedding dim to avoid very small gradients
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        all_context_vecs = attn_weights @ values
        return all_context_vecs
