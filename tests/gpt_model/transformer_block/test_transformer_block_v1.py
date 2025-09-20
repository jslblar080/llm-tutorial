import torch

from llmtutorial.config import Config
from llmtutorial.gpt_model.gpt_model_config import GPTModelConfig
from llmtutorial.gpt_model.transformer_block.transformer_block_v1 import (
    TransformerBlockV1,
)


# pytest -sv tests/gpt_model/transformer_block/test_transformer_block_v1.py
class TestTransformerBlockV1:

    _seed_num = 123
    _batch_size = 2
    _cxt_len = Config().context_length
    _embedding_dim = GPTModelConfig().embedding_dim

    def test_emb_dim_size(self):
        torch.manual_seed(self._seed_num)
        block = TransformerBlockV1()
        x = torch.rand(self._batch_size, self._cxt_len, self._embedding_dim)
        out = block(x)
        assert x.shape == out.shape
