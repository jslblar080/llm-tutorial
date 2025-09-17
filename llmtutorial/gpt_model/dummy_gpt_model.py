import torch.nn as nn

from ..base_gpt_model import BaseGPTModel
from .base_layer_norm import BaseLayerNorm
from .embedder import Embedder
from .gpt_model_config import GPTModelConfig


class DummyGPTModel(BaseGPTModel):

    _dropout: nn.Dropout
    _trf_blocks: nn.Sequential
    _final_layer_norm: BaseLayerNorm
    _output_head: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        gpt_model_config = GPTModelConfig()
        self._dropout = nn.Dropout(gpt_model_config.drop_rate)
        self._trf_blocks = nn.Sequential(
            *[
                GPTModelConfig().dummy_gpt_model_trf_block
                for _ in range(gpt_model_config.num_trf_blocks)
            ]
        )
        GPTModelConfig().dummy_gpt_model_final_layer_norm = (
            gpt_model_config.embedding_dim
        )
        self._final_layer_norm = GPTModelConfig().dummy_gpt_model_final_layer_norm
        self._output_head = nn.Linear(
            gpt_model_config.embedding_dim,
            gpt_model_config.num_embeddings,
            bias=False,
        )

    def forward(self, inputs):
        x = Embedder.input_embeddings(inputs)
        x = self._dropout(x)
        x = self._trf_blocks(x)
        x = self._final_layer_norm(x)
        logits = self._output_head(x)
        return logits
