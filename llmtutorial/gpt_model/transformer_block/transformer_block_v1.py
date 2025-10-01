import copy
import torch.nn as nn

from .base_attention import BaseAttention
from .base_feed_forward import BaseFeedForward
from ..base_layer_norm import BaseLayerNorm
from ..base_transformer_block import BaseTransformerBlock
from ..gpt_model_config import GPTModelConfig


class TransformerBlockV1(BaseTransformerBlock):

    _norm1: BaseLayerNorm
    _att: BaseAttention
    _drop_shortcut: nn.Dropout
    _norm2: BaseLayerNorm
    _ffn: BaseFeedForward

    def __init__(self):
        super().__init__()
        gpt_model_config = GPTModelConfig()
        self._norm1 = copy.deepcopy(
            gpt_model_config.transformer_block_v1_first_layer_norm
        )
        self._att = copy.deepcopy(gpt_model_config.attention)
        self._drop_shortcut = nn.Dropout(gpt_model_config.drop_rate_shortcut)
        self._norm2 = copy.deepcopy(
            gpt_model_config.transformer_block_v1_second_layer_norm
        )
        self._ffn = copy.deepcopy(gpt_model_config.transformer_block_v1_feed_forward)

    def forward(self, x):
        shortcut = x
        x = self._norm1(x)
        x = self._att(x)
        x = self._drop_shortcut(x)
        x = shortcut + x
        shortcut = x
        x = self._norm2(x)
        x = self._ffn(x)
        x = self._drop_shortcut(x)
        x = shortcut + x
        return x
