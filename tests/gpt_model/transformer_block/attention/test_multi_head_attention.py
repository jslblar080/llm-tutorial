from llmtutorial.gpt_model.transformer_block.attention.multi_head_attention import (
    MultiHeadAttention,
)


# pytest -sv tests/gpt_model/transformer_block/attention/test_multi_head_attention.py
class TestMultiHeadAttention:

    _embedding_dim = 768
    _cxt_len = 3
    _drop_rate = 0.1

    def test_num_params(self):
        mha = MultiHeadAttention(
            self._embedding_dim,
            self._embedding_dim,
            self._cxt_len,
            self._drop_rate,
            self._embedding_dim // 64,
        )
        mha_params = sum(p.numel() for p in mha.parameters())
        print(f"\nNumber of parameters in MultiHeadAttention: {mha_params:,}")
