import torch
import torch.nn as nn

from ..base_gpt_model import BaseGPTModel
from ..base_layer_norm import BaseLayerNorm
from ..config import Config
from ..layer_norm.dummy_layer_norm import DummyLayerNorm
from ..transformer_block.dummy_transformer_block import DummyTransformerBlock


class DummyGPTModel(BaseGPTModel):

    _token_embedding_layer: nn.Embedding
    _pos_embedding_layer: nn.Embedding
    _dropout: nn.Dropout
    _trf_blocks: nn.Sequential
    _final_layer_norm: BaseLayerNorm
    _output_head: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        config = Config()
        self._token_embedding_layer = nn.Embedding(
            config.num_embeddings,
            config.embedding_dim,
        )
        self._pos_embedding_layer = nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )
        self._dropout = nn.Dropout(config.drop_rate)
        self._trf_blocks = nn.Sequential(
            *[DummyTransformerBlock() for _ in range(config.num_trf_blocks)]
        )
        self._final_layer_norm = DummyLayerNorm(config.embedding_dim)
        self._output_head = nn.Linear(
            config.embedding_dim,
            config.num_embeddings,
            bias=False,
        )

    def forward(self, inputs):
        batch_size, cxt_len = inputs.shape
        token_embeddings = self._token_embedding_layer(inputs)
        pos_embeddings = self._pos_embedding_layer(
            torch.arange(cxt_len, device=inputs.device)
        )
        x = token_embeddings + pos_embeddings
        x = self._dropout(x)
        x = self._trf_blocks(x)
        x = self._final_layer_norm(x)
        logits = self._output_head(x)
        return logits
