import torch
import torch.nn as nn

from torch import Tensor
from ..base_layer_norm import BaseLayerNorm


class LayerNormV1(BaseLayerNorm):

    _eps: float
    _scale: nn.Parameter
    _shift: nn.Parameter

    def __init__(self, embedding_dim: int, eps=1e-5) -> None:
        super().__init__(embedding_dim)
        self._eps = eps
        self._scale = nn.Parameter(torch.ones(embedding_dim))
        self._shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        biased_var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(biased_var + self._eps)
        return self._scale * norm_x + self._shift
