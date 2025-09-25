import torch

from torch.utils.data import DataLoader
from llmtutorial.config import Config
from llmtutorial.gpt_model.dummy_gpt_model import DummyGPTModel
from llmtutorial.gpt_model.gpt_model_config import GPTModelConfig
from llmtutorial.text_processor import TextProcessor


# pytest -sv tests/gpt_model/test_dummy_gpt_model.py
class TestDummyGPTModel:

    def test_inputs_logits_shape(self):
        torch.manual_seed(123)
        texts = (
            "In the heart of the city stood the old library, a relic from a bygone era.",
            "Its stone walls bore the marks of time, and ivy clung tightly to its facade.",
        )
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
        assert inputs.shape[0] == batch_size
        assert inputs.shape[1] == Config().context_length
        dummy_gpt_model = DummyGPTModel()
        logits = dummy_gpt_model(inputs)
        print("\nShape of logits:", logits.shape)
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == Config().context_length
        assert logits.shape[2] == GPTModelConfig().num_embeddings
