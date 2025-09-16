from ..base_transformer_block import BaseTransformerBlock


class DummyTransformerBlock(BaseTransformerBlock):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
