import torch.nn as nn

from config import Config


class Embedder:

    @staticmethod
    def token_layer() -> nn.Embedding:

        num_embeddings = Config().num_embeddings
        embedding_dim = Config().embedding_dim

        token_embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        print(
            "\nWeight matrix of token embedding layer:\n", token_embedding_layer.weight
        )

        return token_embedding_layer
