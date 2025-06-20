import torch
import torch.nn as nn

from config import Config


class Embedder:

    @staticmethod
    def to_vector(token_ids: list) -> None:

        embedding_layer = nn.Embedding(len(token_ids), Config().embedding_dim)
        print("\nWeight of embedding layer:\n", embedding_layer.weight)
