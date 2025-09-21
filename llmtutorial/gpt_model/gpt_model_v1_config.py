from .base_layer_norm import BaseLayerNorm
from .base_transformer_block import BaseTransformerBlock
from .gpt_model_config import GPTModelConfig
from .layer_norm.layer_norm_v1 import LayerNormV1
from .transformer_block.transformer_block_v1 import TransformerBlockV1
from ..util.singleton_meta import SingletonMeta


class GPTModelV1Config(metaclass=SingletonMeta):

    _gpt_model_v1_trf_block: BaseTransformerBlock
    _gpt_model_v1_final_layer_norm: BaseLayerNorm

    def __init__(self) -> None:
        self._gpt_model_v1_trf_block = TransformerBlockV1()
        self._gpt_model_v1_final_layer_norm = LayerNormV1(
            GPTModelConfig().embedding_dim
        )

    @property
    def gpt_model_v1_trf_block(self):
        return self._gpt_model_v1_trf_block

    @property
    def gpt_model_v1_final_layer_norm(self):
        return self._gpt_model_v1_final_layer_norm
