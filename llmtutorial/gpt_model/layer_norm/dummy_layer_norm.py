from ..base_layer_norm import BaseLayerNorm


class DummyLayerNorm(BaseLayerNorm):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__(embedding_dim)

    def forward(self, x):
        return x
