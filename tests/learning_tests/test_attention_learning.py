import pytest
import torch
import torch.nn as nn


class TestAttentionLearning:

    _inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # x_0
            [0.55, 0.87, 0.66],  # x_1
            [0.57, 0.85, 0.64],  # x_2
            [0.22, 0.58, 0.33],  # x_3
            [0.77, 0.25, 0.10],  # x_4
            [0.05, 0.80, 0.55],  # x_5
        ]
    )

    @pytest.fixture(autouse=True)
    def keep_manual_seed(self):
        torch.manual_seed(123)

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
