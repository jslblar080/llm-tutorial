import torch
import torch.nn as nn

from config import Config


class Embedder:

    @staticmethod
    def to_layer(token_ids: list[int]) -> nn.Embedding:

        embedding_dim = Config().embedding_dim

        embedding_layer = nn.Embedding(len(token_ids), embedding_dim)
        print("\nWeight matrix of embedding layer:\n", embedding_layer.weight)

        return embedding_layer
