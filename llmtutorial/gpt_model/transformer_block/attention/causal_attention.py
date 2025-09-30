import torch
import torch.nn as nn

from torch import Tensor
from ..base_attention import BaseAttention


class CausalAttention(BaseAttention):

    _W_query: nn.Linear
    _W_key: nn.Linear
    _W_value: nn.Linear
    _mask: Tensor
    _dropout: nn.Dropout

    def __init__(
        self,
        seed_num: int,
        d_in: int,
        d_out: int,
        cxt_len: int,
        dr: float,
        qkv_bias=False,
    ):
        super().__init__()
        torch.manual_seed(seed_num)
        self._W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # buffers automatically move to appropriate device (CPU or GPU)
        self.register_buffer(
            "_mask", torch.triu(torch.ones(cxt_len, cxt_len), diagonal=1)
        )
        self._dropout = nn.Dropout(dr)

    def forward(self, x):
        batch_size, cxt_len, d_in = x.shape
        queries = self._W_query(x)
        keys = self._W_key(x)
        values = self._W_value(x)
        attn_scores: Tensor = queries @ keys.transpose(
            1, 2  # 0: batch_size, 1: cxt_len, 2: d_out
        )
        attn_scores.masked_fill_(self._mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self._dropout(attn_weights)
        all_context_vecs = attn_weights @ values
        return all_context_vecs
