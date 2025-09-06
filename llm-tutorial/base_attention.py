import torch.nn as nn


class BaseAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError(
            f'{type(self).__name__} is missing the implementation of "forward" function'
        )
