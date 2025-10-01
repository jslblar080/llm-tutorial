import torch
import torch.nn as nn

from torch import Tensor
from .gpt_model_config import GPTModelConfig


class Embedder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._token_emb_layer = nn.Embedding(
            GPTModelConfig().num_embeddings, GPTModelConfig().embedding_dim
        )
        from ..config import Config

        self._pos_emb_layer = nn.Embedding(
            Config().context_length, GPTModelConfig().embedding_dim
        )

    def forward(self, x):
        batch_size, cxt_len = x.shape
        token_embeddings = self._token_emb_layer(x)
        pos_embeddings = self._pos_emb_layer(torch.arange(cxt_len, device=x.device))
        return token_embeddings + pos_embeddings

    @staticmethod
    def _token_layer() -> nn.Embedding:
        token_embedding_layer = nn.Embedding(
            GPTModelConfig().num_embeddings, GPTModelConfig().embedding_dim
        )
        return token_embedding_layer

    @classmethod
    def tok_emb_weight(cls, verbose=False) -> Tensor:
        token_embedding_layer = cls._token_layer()
        if verbose:
            print(
                "\nWeight matrix of token embedding layer:\n",
                token_embedding_layer.weight,
            )
        return token_embedding_layer.weight

    @staticmethod
    def _pos_layer(cxt_len: int) -> nn.Embedding:
        pos_embedding_layer = nn.Embedding(cxt_len, GPTModelConfig().embedding_dim)
        return pos_embedding_layer

    @classmethod
    def input_embeddings(cls, inputs: Tensor) -> Tensor:
        batch_size, cxt_len = inputs.shape
        token_embedding_layer = cls._token_layer()
        token_embeddings = token_embedding_layer(inputs)
        pos_embedding_layer = cls._pos_layer(cxt_len)
        pos_embeddings = pos_embedding_layer(
            torch.arange(cxt_len, device=inputs.device)
        )
        return token_embeddings + pos_embeddings
