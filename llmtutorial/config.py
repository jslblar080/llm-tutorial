from typing import Tuple
from .base_dataset import BaseDataset
from .dataset.gpt_dataset_v1 import GPTDatasetV1
from .util.singleton_meta import SingletonMeta


class Config(metaclass=SingletonMeta):

    _texts: Tuple[str, ...]
    _context_length: int
    _dataset: BaseDataset
    _encoding: str

    def __init__(self) -> None:
        self._texts = ("assets", "sample.txt")
        self._context_length = 3
        self._encoding = "o200k_base"  # token ID of <|endoftext|>: 199999

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

    @dataset.setter
    def dataset(self, token_ids: list[int]):
        self._dataset = GPTDatasetV1(
            token_ids, max_length=self._context_length, stride=self._context_length
        )
