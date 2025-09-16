from .base_layer_norm import BaseLayerNorm
from .base_transformer_block import BaseTransformerBlock
from .layer_norm.dummy_layer_norm import DummyLayerNorm
from .transformer_block.dummy_transformer_block import DummyTransformerBlock
from ..util.singleton_meta import SingletonMeta


class GPTModelConfig(metaclass=SingletonMeta):

    _dummy_gpt_model_final_layer_norm: BaseLayerNorm
    _dummy_gpt_model_trf_block: BaseTransformerBlock

    def __init__(self) -> None:
        self._dummy_gpt_model_trf_block = DummyTransformerBlock()

    @property
    def dummy_gpt_model_final_layer_norm(self):
        return self._dummy_gpt_model_final_layer_norm

    @property
    def dummy_gpt_model_trf_block(self):
        return self._dummy_gpt_model_trf_block

    @dummy_gpt_model_final_layer_norm.setter
    def dummy_gpt_model_final_layer_norm(self, embedding_dim: int):
        self._dummy_gpt_model_final_layer_norm = DummyLayerNorm(embedding_dim)
