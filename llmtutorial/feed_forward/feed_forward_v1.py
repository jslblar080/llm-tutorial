import torch.nn as nn

from ..base_feed_forward import BaseFeedForward


class FeedForwardV1(BaseFeedForward):

    _layers: nn.Sequential

    def __init__(self, embedding_dim: int, activation_function: nn.Module) -> None:
        super().__init__(embedding_dim)
        self._layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            activation_function,
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self._layers(x)
