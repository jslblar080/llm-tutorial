from torch import Tensor
from typing import Tuple
from attention.causal_attention import CausalAttention
from attention.multi_head_attention import MultiHeadAttention
from attention.self_attention import SelfAttention
from attention.simplified_self_attention import SimplifiedSelfAttention
from base_attention import BaseAttention
from base_dataset import BaseDataset
from dataset.gpt_dataset_v1 import GPTDatasetV1
from util.singleton_meta import SingletonMeta


class Config(metaclass=SingletonMeta):

    _texts: Tuple[str, ...]
    _context_length: int
    _dataset: BaseDataset
    _encoding: str
    _num_embeddings: int
    _embedding_dim: int
    _attention: BaseAttention

    def __init__(self) -> None:
        self._texts = (
            "In the heart of the city stood the old library, a relic from a bygone era.",
            "Its stone walls bore the marks of time, and ivy clung tightly to its facade.",
        )
        self._context_length = 3
        self._encoding = "o200k_base"  # token ID of <|endoftext|>: 199999
        self._num_embeddings = (
            200000  # TODO: Update automatically according to _encoding
        )
        self._embedding_dim = 256

    @property
    def texts(self):
        return self._texts

    @property
    def context_length(self):
        return self._context_length

    @property
    def dataset(self):
        return self._dataset

    @property
    def encoding(self):
        return self._encoding

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def attention(self):
        return self._attention

    @dataset.setter
    def dataset(self, token_ids: list[int]):
        self._dataset = GPTDatasetV1(
            token_ids, max_length=self._context_length, stride=self._context_length
        )

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
