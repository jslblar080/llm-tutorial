from ..base_layer_norm import BaseLayerNorm


class DummyLayerNorm(BaseLayerNorm):

    def __init__(self, normalized_shape: int, eps=1e-5) -> None:
        super().__init__(normalized_shape)

    def forward(self, x):
        return x
