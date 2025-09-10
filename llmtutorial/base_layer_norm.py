import torch.nn as nn


class BaseLayerNorm(nn.Module):

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError(
            f'{type(self).__name__} is missing the implementation of "forward" function'
        )
