import torch

from llmtutorial.config import Config
from llmtutorial.gpt_model.gpt_model_config import GPTModelConfig
from llmtutorial.gpt_model.transformer_block.transformer_block_v1 import (
    TransformerBlockV1,
)


# pytest -sv tests/gpt_model/transformer_block/test_transformer_block_v1.py
class TestTransformerBlockV1:

    def test_emb_dim_size(self):
        block = TransformerBlockV1()
        x = torch.rand(
            Config().batch_size, Config().context_length, GPTModelConfig().embedding_dim
        )
        out = block(x)
        assert x.shape == out.shape
