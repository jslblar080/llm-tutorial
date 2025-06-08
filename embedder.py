import torch
import torch.nn as nn

from config import Config


class Embedder:

    @staticmethod
    def to_vector(word2idx_dict: dict):

        embed_layer = nn.Embedding(len(word2idx_dict), Config().embedding_dim)
        embeddings = embed_layer(torch.tensor(list(word2idx_dict.values()))).unsqueeze(
            0
        )
        print("input_embeddings.shape:", embeddings.shape)
