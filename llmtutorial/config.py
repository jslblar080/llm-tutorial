from .base_dataset import BaseDataset
from .dataset.gpt_dataset_v1 import GPTDatasetV1
from .util.singleton_meta import SingletonMeta


class Config(metaclass=SingletonMeta):

    _texts: tuple[str, ...]
    _train_ratio: float
    _context_length: int
    _dataset: BaseDataset
    _encoding: str

    def __init__(self) -> None:
        self.initialize()

    def initialize(self) -> None:
        self._texts = ("assets", "sample.txt")
        self._train_ratio = 0.9
        self._context_length = 3
        self._encoding = "o200k_base"  # token ID of <|endoftext|>: 199999

    @property
    def texts(self):
        return self._texts

    @property
    def train_ratio(self):
        return self._train_ratio

    @property
    def context_length(self):
        return self._context_length

    @property
    def dataset(self):
        return self._dataset

    @property
    def encoding(self):
        return self._encoding

    @texts.setter
    def texts(self, text_path: tuple[str, ...]):
        self._texts = text_path

    @train_ratio.setter
    def train_ratio(self, train_ratio: float):
        self._train_ratio = train_ratio

    @context_length.setter
    def context_length(self, cxt_len: int):
        self._context_length = cxt_len

    @encoding.setter
    def encoding(self, encoding: str):
        self._encoding = encoding

    @dataset.setter
    def dataset(self, token_ids: list[int]):
        self._dataset = GPTDatasetV1(
            token_ids, max_length=self._context_length, stride=self._context_length
        )
