from typing import Tuple
from util.singleton_meta import SingletonMeta


class Config(metaclass=SingletonMeta):

    _texts: Tuple[str, ...]
    _encoding: str
    _embedding_dim: int

    def __init__(self) -> None:
        self._texts = (
            "In the heart of the city stood the old library, a relic from a bygone era.",
            "Its stone walls bore the marks of time, and ivy clung tightly to its facade.",
        )
        self._encoding = "o200k_base"  # token ID of <|endoftext|>: 199999
        self._embedding_dim = 3

    @property
    def texts(self):
        return self._texts

    @property
    def encoding(self):
        return self._encoding

    @property
    def embedding_dim(self):
        return self._embedding_dim
