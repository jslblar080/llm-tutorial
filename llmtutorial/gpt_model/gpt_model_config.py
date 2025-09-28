import tiktoken

from .base_layer_norm import BaseLayerNorm
from .base_transformer_block import BaseTransformerBlock
from .layer_norm.dummy_layer_norm import DummyLayerNorm
from .layer_norm.layer_norm_v1 import LayerNormV1
from .transformer_block.attention.causal_attention import CausalAttention
from .transformer_block.attention.multi_head_attention import (
    MultiHeadAttention,
)
from .transformer_block.attention.self_attention import SelfAttention
from .transformer_block.attention.simplified_self_attention import (
    SimplifiedSelfAttention,
)
from .transformer_block.base_attention import BaseAttention
from .transformer_block.base_feed_forward import BaseFeedForward
from .transformer_block.dummy_transformer_block import DummyTransformerBlock
from .transformer_block.feed_forward.activation_function.GELU_approx import GELUApprox
from .transformer_block.feed_forward.feed_forward_v1 import FeedForwardV1
from ..util.singleton_meta import SingletonMeta


class GPTModelConfig(metaclass=SingletonMeta):

    _num_embeddings: int
    _embedding_dim: int
    _drop_rate_emb: float
    _drop_rate_attn: float
    _drop_rate_shortcut: float
    _num_trf_blocks: int
    _attention: BaseAttention

    _transformer_block_v1_first_layer_norm: BaseLayerNorm
    _transformer_block_v1_second_layer_norm: BaseLayerNorm
    _transformer_block_v1_feed_forward: BaseFeedForward
    _dummy_gpt_model_final_layer_norm: BaseLayerNorm
    _dummy_gpt_model_trf_block: BaseTransformerBlock

    def __init__(self) -> None:
        self.initialize()

    def initialize(self) -> None:
        from ..config import Config

        self._num_embeddings = (
            tiktoken.get_encoding(Config().encoding).encode_single_token(
                "<|endoftext|>"
            )
            + 1
        )
        self._embedding_dim = 64 * 4
        self._drop_rate_emb = 0.1
        self._drop_rate_attn = 0.1
        self._drop_rate_shortcut = 0.1
        self._num_trf_blocks = 12
        assert self._embedding_dim % 64 == 0, "_embedding_dim must be divisible by 64"
        self._attention = MultiHeadAttention(
            self.embedding_dim,
            self.embedding_dim,
            Config().context_length,
            self._drop_rate_attn,
            self._embedding_dim // 64,
        )

        self._transformer_block_v1_first_layer_norm = LayerNormV1(self._embedding_dim)
        self._transformer_block_v1_second_layer_norm = LayerNormV1(self._embedding_dim)
        self._transformer_block_v1_feed_forward = FeedForwardV1(
            self._embedding_dim, GELUApprox()
        )
        self._dummy_gpt_model_final_layer_norm = DummyLayerNorm(self._embedding_dim)
        self._dummy_gpt_model_trf_block = DummyTransformerBlock()

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def drop_rate_emb(self):
        return self._drop_rate_emb

    @property
    def drop_rate_attn(self):
        return self._drop_rate_attn

    @property
    def drop_rate_shortcut(self):
        return self._drop_rate_shortcut

    @property
    def num_trf_blocks(self):
        return self._num_trf_blocks

    @property
    def attention(self):
        return self._attention

    @property
    def transformer_block_v1_first_layer_norm(self):
        return self._transformer_block_v1_first_layer_norm

    @property
    def transformer_block_v1_second_layer_norm(self):
        return self._transformer_block_v1_second_layer_norm

    @property
    def transformer_block_v1_feed_forward(self):
        return self._transformer_block_v1_feed_forward

    @property
    def dummy_gpt_model_final_layer_norm(self):
        return self._dummy_gpt_model_final_layer_norm

    @property
    def dummy_gpt_model_trf_block(self):
        return self._dummy_gpt_model_trf_block

    @embedding_dim.setter
    def embedding_dim(self, embedding_dim: int):
        assert embedding_dim > 0, "_embedding_dim must be positive"
        self._embedding_dim = embedding_dim

    @num_trf_blocks.setter
    def num_trf_blocks(self, num_trf_blocks: int):
        assert num_trf_blocks > 0, "_num_trf_blocks must be positive"
        self._num_trf_blocks = num_trf_blocks
