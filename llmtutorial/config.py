from .base_dataset import BaseDataset
from .base_gpt_model import BaseGPTModel
from .dataset.gpt_dataset_v1 import GPTDatasetV1
from .gpt_model.gpt_model_v1 import GPTModelV1
from .util.singleton_meta import SingletonMeta


class Config(metaclass=SingletonMeta):

    _texts: tuple[str, ...]
    _train_ratio: float
    _seed_num: int
    _batch_size: int
    _num_workers: int
    _context_length: int
    _encoding: str
    _dataset: BaseDataset
    _gpt_model: BaseGPTModel

    def __init__(self) -> None:
        self.initialize()

    def initialize(self) -> None:
        self._texts = ("assets", "sample.txt")
        self._train_ratio = 0.9
        self._seed_num = 123
        self._batch_size = 3
        self._num_workers = 0
        self._context_length = 3
        self._encoding = "o200k_base"  # token ID of <|endoftext|>: 199999
        self._gpt_model = GPTModelV1()

    @property
    def texts(self):
        return self._texts

    @property
    def train_ratio(self):
        return self._train_ratio

    @property
    def seed_num(self):
        return self._seed_num

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_workers(self):
        return self._num_workers

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
    def gpt_model(self):
        return self._gpt_model

    @texts.setter
    def texts(self, text_path: tuple[str, ...]):
        self._texts = text_path

    @train_ratio.setter
    def train_ratio(self, train_ratio: float):
        self._train_ratio = train_ratio

    @seed_num.setter
    def seed_num(self, seed_num: int):
        self._seed_num = seed_num

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    @num_workers.setter
    def num_workers(self, num_workers: int):
        self._num_workers = num_workers

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
