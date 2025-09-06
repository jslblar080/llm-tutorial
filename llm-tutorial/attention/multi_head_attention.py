import torch
import torch.nn as nn

from torch import Tensor
from base_attention import BaseAttention


class MultiHeadAttention(BaseAttention):

    _n_head: int
    _d_head: int
    _W_query: nn.Linear
    _W_key: nn.Linear
    _W_value: nn.Linear
    _mask: Tensor
    _dropout: nn.Dropout
    _out_proj: nn.Identity | nn.Linear

    def __init__(
        self,
        d_in: int,
        d_out: int,
        cxt_len: int,
        dr: float,
        n_head: int,
        qkv_bias=False,
        out_proj=True,
    ):
        super().__init__()
        assert d_out % n_head == 0, "d_out must be divisible by n_head"
        self._n_head = n_head
        self._d_head = d_out // n_head
        self._W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self._W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer(
            "_mask", torch.triu(torch.ones(cxt_len, cxt_len), diagonal=1)
        )
        self._dropout = nn.Dropout(dr)
        self._out_proj = nn.Linear(d_out, d_out) if out_proj else nn.Identity()

    def forward(self, x):
        batch_size, cxt_len, d_in = x.shape
        # unlike reshape, view requires contiguous tensor
        # output tensor of nn.Linear is always contiguous
        queries = (
            self._W_query(x)
            .view(batch_size, cxt_len, self._n_head, self._d_head)
            .transpose(1, 2)
        )
        keys = (
            self._W_key(x)
            .view(batch_size, cxt_len, self._n_head, self._d_head)
            .transpose(1, 2)
        )
        values = (
            self._W_value(x)
            .view(batch_size, cxt_len, self._n_head, self._d_head)
            .transpose(1, 2)
        )
        attn_scores: Tensor = queries @ keys.transpose(
            2, 3  # 0: batch_size, 1: self._n_head, 2: cxt_len, 3: self._d_head
        )
        attn_scores.masked_fill_(self._mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self._dropout(attn_weights)
        all_context_vecs = (
            (attn_weights @ values)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, cxt_len, self._n_head * self._d_head)
        )
        # optional linear projection
        all_context_vecs = self._out_proj(all_context_vecs)
        return all_context_vecs
