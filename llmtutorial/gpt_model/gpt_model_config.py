from torch import Tensor
from .base_layer_norm import BaseLayerNorm
from .base_transformer_block import BaseTransformerBlock
from .layer_norm.dummy_layer_norm import DummyLayerNorm
from .transformer_block.attention.causal_attention import CausalAttention
from .transformer_block.attention.multi_head_attention import (
    MultiHeadAttention,
)
from .transformer_block.attention.self_attention import SelfAttention
from .transformer_block.attention.simplified_self_attention import (
    SimplifiedSelfAttention,
)
from .transformer_block.base_attention import BaseAttention
from .transformer_block.dummy_transformer_block import DummyTransformerBlock
from ..util.singleton_meta import SingletonMeta


class GPTModelConfig(metaclass=SingletonMeta):

    _num_embeddings: int
    _embedding_dim: int
    _drop_rate: float
    _num_trf_blocks: int
    _attention: BaseAttention

    _dummy_gpt_model_final_layer_norm: BaseLayerNorm
    _dummy_gpt_model_trf_block: BaseTransformerBlock

    def __init__(self) -> None:
        self._num_embeddings = (
            200000  # TODO: Update automatically according to Config().encoding
        )
        self._embedding_dim = 256
        self._drop_rate = 0.1
        self._num_trf_blocks = 12

        self._dummy_gpt_model_trf_block = DummyTransformerBlock()

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def drop_rate(self):
        return self._drop_rate

    @property
    def num_trf_blocks(self):
        return self._num_trf_blocks

    @property
    def attention(self):
        return self._attention

    @property
    def dummy_gpt_model_final_layer_norm(self):
        return self._dummy_gpt_model_final_layer_norm

    @property
    def dummy_gpt_model_trf_block(self):
        return self._dummy_gpt_model_trf_block

    @attention.setter
    def attention(self, batch_embeddings: Tensor):
        assert batch_embeddings.ndim == 3, "batch_embeddings must be 3D"
        assert self._embedding_dim % 64 == 0, "_embedding_dim must be divisible by 64"
        self._attention = MultiHeadAttention(
            batch_embeddings.shape[2],
            batch_embeddings.shape[2],
            batch_embeddings.shape[1],
            0.1,
            self._embedding_dim // 64,
        )

    @dummy_gpt_model_final_layer_norm.setter
    def dummy_gpt_model_final_layer_norm(self, embedding_dim: int):
        self._dummy_gpt_model_final_layer_norm = DummyLayerNorm(embedding_dim)
