import torch.nn as nn

from .base_layer_norm import BaseLayerNorm
from .gpt_model_config import GPTModelConfig
from .layer_norm.layer_norm_v1 import LayerNormV1
from .transformer_block.transformer_block_v1 import TransformerBlockV1
from ..util.singleton_meta import SingletonMeta


class GPTModelV1Config(metaclass=SingletonMeta):

    _gpt_model_v1_trf_blocks: nn.Sequential
    _gpt_model_v1_final_layer_norm: BaseLayerNorm

    def __init__(self) -> None:
        self.initialize()

    def initialize(self) -> None:
        self._gpt_model_v1_trf_blocks = nn.Sequential(
            *[TransformerBlockV1() for _ in range(GPTModelConfig().num_trf_blocks)]
        )
        self._gpt_model_v1_final_layer_norm = LayerNormV1(
            GPTModelConfig().embedding_dim
        )

    @property
    def gpt_model_v1_trf_blocks(self):
        return self._gpt_model_v1_trf_blocks

    @property
    def gpt_model_v1_final_layer_norm(self):
        return self._gpt_model_v1_final_layer_norm
