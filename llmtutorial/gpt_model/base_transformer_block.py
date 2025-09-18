import torch.nn as nn

from ..util.singleton_meta import SingletonMeta


class BaseTransformerBlock(nn.Module, metaclass=SingletonMeta):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError(
            f'{type(self).__name__} is missing the implementation of "forward" function'
        )
