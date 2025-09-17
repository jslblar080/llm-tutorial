import torch
import torch.nn as nn

from torch import Tensor
from ..config import Config


class Embedder:

    @staticmethod
    def _token_layer() -> nn.Embedding:

        token_embedding_layer = nn.Embedding(
            Config().num_embeddings, Config().embedding_dim
        )
        # print(
        #     "\nWeight matrix of token embedding layer:\n", token_embedding_layer.weight
        # )

        return token_embedding_layer

    @staticmethod
    def _pos_layer() -> nn.Embedding:

        pos_embedding_layer = nn.Embedding(
            Config().context_length, Config().embedding_dim
        )

        return pos_embedding_layer

    @classmethod
    def input_embeddings(cls, inputs: Tensor) -> Tensor:

        token_embedding_layer = cls._token_layer()
        token_embeddings = token_embedding_layer(inputs)

        pos_embedding_layer = cls._pos_layer()
        pos_embeddings = pos_embedding_layer(
            torch.arange(Config().context_length, device=inputs.device)
        )

        return token_embeddings + pos_embeddings
