import torch
import torch.nn as nn

from config import Config


class Embedder:

    @staticmethod
    def to_vector(token_ids: list[int]) -> None:

        embedding_dim = Config().embedding_dim

        embedding_layer = nn.Embedding(len(token_ids), embedding_dim)
        print("\nWeight of embedding layer:\n", embedding_layer.weight)
