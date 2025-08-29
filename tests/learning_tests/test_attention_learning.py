import torch


class TestAttentionLearning:

    def test_simplified_self_attention(self):
        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],  # x_0
                [0.55, 0.87, 0.66],  # x_1 (query)
                [0.57, 0.85, 0.64],  # x_2
                [0.22, 0.58, 0.33],  # x_3
                [0.77, 0.25, 0.10],  # x_4
                [0.05, 0.80, 0.55],  # x_5
            ]
        )
        attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
        """
        for i, x_i in enumerate(inputs):
            for j, x_j in enumerate(inputs):
                attn_scores[i, j] = torch.dot(x_i, x_j)
        """
        attn_scores = inputs @ inputs.T  # matrix multiplication
        print("\nAttention scores:\n", attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print("Attention weights:\n", attn_weights)
        for sum in attn_weights.sum(dim=-1):
            torch.testing.assert_close(sum.item(), 1.0)
        all_context_vecs = attn_weights @ inputs
        print("All context vectors:\n", all_context_vecs)
