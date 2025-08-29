import torch


class TestAttentionLearning:

    def test_simplified_self_attention(self):
        inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89],  # x_1
                [0.55, 0.87, 0.66],  # x_2 (query)
                [0.57, 0.85, 0.64],  # x_3
                [0.22, 0.58, 0.33],  # x_4
                [0.77, 0.25, 0.10],  # x_5
                [0.05, 0.80, 0.55],  # x_6
            ]
        )
        query = inputs[1]
        attn_scores_query = torch.empty(inputs.shape[0])
        for i, x_i in enumerate(inputs):
            attn_scores_query[i] = torch.dot(x_i, query)
        print("\nAttention scores:", attn_scores_query)
        attn_weights_query = torch.softmax(attn_scores_query, dim=0)
        print("Attention weights:", attn_weights_query)
        attn_weights_query_sum = attn_weights_query.sum().item()
        print("Sum of attention weights:", attn_weights_query_sum)
        torch.testing.assert_close(attn_weights_query_sum, 1.0)
        context_vec_query = torch.zeros(query.shape)
        for i, x_i in enumerate(inputs):
            context_vec_query += attn_weights_query[i] * x_i
        print(context_vec_query)
