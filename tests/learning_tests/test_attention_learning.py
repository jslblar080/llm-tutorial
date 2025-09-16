import pytest
import torch
import torch.nn as nn

from torch import Tensor


# pytest -sv tests/learning_tests/test_attention_learning.py
class TestAttentionLearning:

    class SimplifiedSelfAttention(nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x):
            attn_scores = torch.empty(x.shape[0], x.shape[0])
            attn_scores = x @ x.T
            attn_weights = torch.softmax(attn_scores, dim=-1)
            all_context_vecs = attn_weights @ x
            return all_context_vecs

    class SelfAttentionParameter(nn.Module):

        _W_query: nn.Parameter
        _W_key: nn.Parameter
        _W_value: nn.Parameter

        def __init__(self, d_in: int, d_out: int, seed_num: int):
            super().__init__()
            torch.manual_seed(seed_num)
            self._W_query = nn.Parameter(torch.rand(d_in, d_out))
            self._W_key = nn.Parameter(torch.rand(d_in, d_out))
            self._W_value = nn.Parameter(torch.rand(d_in, d_out))

        def set_W_query(self, new_W_query: nn.Parameter) -> None:
            self._W_query = new_W_query

        def set_W_key(self, new_W_key: nn.Parameter) -> None:
            self._W_key = new_W_key

        def set_W_value(self, new_W_value: nn.Parameter) -> None:
            self._W_value = new_W_value

        def forward(self, x):
            queries = x @ self._W_query
            keys = x @ self._W_key
            values = x @ self._W_value
            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            all_context_vecs = attn_weights @ values
            return all_context_vecs

    class SelfAttentionLinear(nn.Module):

        _W_query: nn.Linear
        _W_key: nn.Linear
        _W_value: nn.Linear

        def __init__(self, d_in: int, d_out: int, seed_num: int, qkv_bias=False):
            super().__init__()
            torch.manual_seed(seed_num)
            self._W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self._W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self._W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        @property
        def W_query(self):
            return self._W_query

        @property
        def W_key(self):
            return self._W_key

        @property
        def W_value(self):
            return self._W_value

        def forward(self, x):
            queries = self._W_query(x)
            keys = self._W_key(x)
            values = self._W_value(x)
            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            all_context_vecs = attn_weights @ values
            return all_context_vecs

    class CausalAttention(nn.Module):

        _W_query: nn.Linear
        _W_key: nn.Linear
        _W_value: nn.Linear
        _mask: Tensor
        _dropout: nn.Dropout

        def __init__(
            self,
            d_in: int,
            d_out: int,
            cxt_len: int,
            dr: float,
            seed_num: int,
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

    class MultiHeadAttention(nn.Module):

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
            seed_num: int,
            qkv_bias=False,
            out_proj=True,
        ):
            super().__init__()
            torch.manual_seed(seed_num)
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

    _arr = [
        [0.43, 0.15, 0.89],  # x_0
        [0.55, 0.87, 0.66],  # x_1
        [0.57, 0.85, 0.64],  # x_2
        [0.22, 0.58, 0.33],  # x_3
        [0.77, 0.25, 0.10],  # x_4
        [0.05, 0.80, 0.55],  # x_5
    ]

    _inputs = torch.tensor(_arr)

    _batch = torch.tensor(
        [
            _arr,
            _arr,
        ]
    )

    _seed_num = 123

    @pytest.fixture(autouse=True)
    def keep_manual_seed(self):
        torch.manual_seed(self._seed_num)

    def test_simplified_self_attention(self):
        attn_scores = torch.empty(self._inputs.shape[0], self._inputs.shape[0])
        """
        for i, x_i in enumerate(self._inputs):
            for j, x_j in enumerate(self._inputs):
                attn_scores[i, j] = torch.dot(x_i, x_j)
        """
        attn_scores = self._inputs @ self._inputs.T  # matrix multiplication
        print("\nAttention scores:\n", attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print("Attention weights:\n", attn_weights)
        for sum in attn_weights.sum(dim=-1):
            torch.testing.assert_close(sum.item(), 1.0)
        all_context_vecs = attn_weights @ self._inputs
        print("All context vectors:\n", all_context_vecs)
        ssa = self.SimplifiedSelfAttention()
        torch.testing.assert_close(all_context_vecs, ssa(self._inputs))

    def test_self_attention_nn_parameter(self):
        d_in = self._inputs.shape[1]
        d_out = self._inputs.shape[1]
        W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        queries = self._inputs @ W_query
        keys = self._inputs @ W_key
        values = self._inputs @ W_value
        attn_scores = queries @ keys.T
        print("\nAttention scores:\n", attn_scores)
        # normalize by embedding dim to avoid large dot products (very small gradients)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        print("Attention weights:\n", attn_weights)
        for sum in attn_weights.sum(dim=-1):
            torch.testing.assert_close(sum.item(), 1.0)
        all_context_vecs = attn_weights @ values
        print("All context vectors:\n", all_context_vecs)
        sap = self.SelfAttentionParameter(
            self._inputs.shape[1], self._inputs.shape[1], self._seed_num
        )
        torch.testing.assert_close(all_context_vecs, sap(self._inputs))

    def test_self_attention_nn_linear(self):
        d_in = self._inputs.shape[1]
        d_out = self._inputs.shape[1]
        qkv_bias = False
        # nn.Linear use optimized weight initialization scheme
        W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        queries = W_query(self._inputs)
        keys = W_key(self._inputs)
        values = W_value(self._inputs)
        attn_scores = queries @ keys.T
        print("\nAttention scores:\n", attn_scores)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        print("Attention weights:\n", attn_weights)
        for sum in attn_weights.sum(dim=-1):
            torch.testing.assert_close(sum.item(), 1.0)
        all_context_vecs = attn_weights @ values
        print("All context vectors:\n", all_context_vecs)
        sal = self.SelfAttentionLinear(
            self._inputs.shape[1], self._inputs.shape[1], self._seed_num
        )
        torch.testing.assert_close(all_context_vecs, sal(self._inputs))
        sap = self.SelfAttentionParameter(
            self._inputs.shape[1], self._inputs.shape[1], self._seed_num
        )
        sap.set_W_query(nn.Parameter(sal.W_query.weight.T))
        sap.set_W_key(nn.Parameter(sal.W_key.weight.T))
        sap.set_W_value(nn.Parameter(sal.W_value.weight.T))
        torch.testing.assert_close(sal(self._inputs), sap(self._inputs))

    def test_causal_attention(self):
        d_in = self._batch.shape[2]
        d_out = self._batch.shape[2]
        cxt_len = self._batch.shape[1]
        dr = 0.0
        ca = self.CausalAttention(d_in, d_out, cxt_len, dr, self._seed_num)
        all_context_vecs = ca(self._batch)
        print("\nall_context_vecs.shape:", all_context_vecs.shape)
        sal = self.SelfAttentionLinear(d_in, d_out, self._seed_num)
        # no masking on last row with dropout rate 0.0
        torch.testing.assert_close(all_context_vecs[0][-1], sal(self._inputs)[-1])

    def test_single_head_attention(self):
        d_in = self._batch.shape[2]
        d_out = self._batch.shape[2]
        cxt_len = self._batch.shape[1]
        dr = 0.0
        n_head = 1
        sha = self.MultiHeadAttention(
            d_in, d_out, cxt_len, dr, n_head, self._seed_num, out_proj=False
        )
        ca = self.CausalAttention(d_in, d_out, cxt_len, dr, self._seed_num)
        torch.testing.assert_close(sha(self._batch), ca(self._batch))
