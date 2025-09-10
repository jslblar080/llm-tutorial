import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from llmtutorial.base_gpt_model import BaseGPTModel
from llmtutorial.base_layer_norm import BaseLayerNorm
from llmtutorial.layer_norm.dummy_layer_norm import DummyLayerNorm
from llmtutorial.config import Config
from llmtutorial.text_processor import TextProcessor
from llmtutorial.transformer_block.dummy_transformer_block import DummyTransformerBlock


# pytest -sv tests/learning_tests/test_gpt_model_learning.py
class TestGPTModelLearning:

    class DummyGPTModel(BaseGPTModel):

        _token_embedding_layer: nn.Embedding
        _pos_embedding_layer: nn.Embedding
        _dropout: nn.Dropout
        _trf_blocks: nn.Sequential
        _final_layer_norm: BaseLayerNorm
        _output_head: nn.Linear

        def __init__(self) -> None:
            super().__init__()
            config = Config()
            self._token_embedding_layer = nn.Embedding(
                config.num_embeddings,
                config.embedding_dim,
            )
            self._pos_embedding_layer = nn.Embedding(
                config.context_length,
                config.embedding_dim,
            )
            self._dropout = nn.Dropout(config.drop_rate)
            self._trf_blocks = nn.Sequential(
                *[DummyTransformerBlock() for _ in range(config.num_trf_blocks)]
            )
            self._final_layer_norm = DummyLayerNorm(config.embedding_dim)
            self._output_head = nn.Linear(
                config.embedding_dim,
                config.num_embeddings,
                bias=False,
            )

        def forward(self, inputs):
            batch_size, cxt_len = inputs.shape
            token_embeddings = self._token_embedding_layer(inputs)
            pos_embeddings = self._pos_embedding_layer(
                torch.arange(cxt_len, device=inputs.device)
            )
            x = token_embeddings + pos_embeddings
            x = self._dropout(x)
            x = self._trf_blocks(x)
            x = self._final_layer_norm(x)
            logits = self._output_head(x)
            return logits

    def test_dummy_gpt_model(self):
        torch.manual_seed(123)
        texts = Config().texts
        text_processor = TextProcessor()
        token_ids = text_processor.tokenize(
            texts, verbose=False, id_end=True, pair=False
        )
        Config().dataset = token_ids
        dataset = Config().dataset
        batch_size = 3
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        print("\nInputs:\n", inputs)
        dummy_gpt_model = self.DummyGPTModel()
        logits = dummy_gpt_model(inputs)
        print("\nShape of logits:", logits.shape)
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == Config().context_length
        assert logits.shape[2] == Config().num_embeddings
