import tiktoken
import torch

from torch.utils.data import DataLoader
from llmtutorial.config import Config
from llmtutorial.gpt_model.embedder import Embedder
from llmtutorial.gpt_model.gpt_model_config import GPTModelConfig
from llmtutorial.gpt_model.gpt_model_v1 import GPTModelV1
from llmtutorial.text_processor import TextProcessor


# pytest -sv tests/gpt_model/test_gpt_model_v1.py
class TestDummyGPTModel:

    def test_inputs_logits_shape(self):
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
        assert inputs.shape[0] == batch_size
        assert inputs.shape[1] == Config().context_length
        gpt_model_v1 = GPTModelV1()
        logits = gpt_model_v1(inputs)
        print("\nShape of logits:", logits.shape)
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == Config().context_length
        assert logits.shape[2] == GPTModelConfig().num_embeddings

    def test_num_params_memory_size(self):
        torch.manual_seed(123)
        gpt_model_v1 = GPTModelV1()
        total_params = sum(p.numel() for p in gpt_model_v1.parameters())
        print(f"\nTotal number of parameters: {total_params:,}")
        print("Token embedding layer shape:", Embedder.tok_emb_weight().shape)
        print("Output layer shape:", gpt_model_v1.output_head.weight.shape)
        total_params_gpt_model_v1 = total_params - sum(
            p.numel() for p in gpt_model_v1.output_head.parameters()
        )
        num_embeddings, embedding_dim = Embedder.tok_emb_weight().shape
        assert (
            total_params_gpt_model_v1 == total_params - num_embeddings * embedding_dim
        )
        print(
            f"Number of trainable parameters "
            f"considering weight tying: {total_params_gpt_model_v1:,} = {total_params:,} - {num_embeddings:,} * {embedding_dim:,}"
        )
        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB")

    def test_gpt_model_config(self):
        name_embdim_numtrf = (
            ("GPT-small", 768, 12),
            ("GPT-medium", 1024, 24),
            ("GPT-large", 1280, 36),
            ("GPT-XL", 1600, 48),
        )
        gpt_model_config = GPTModelConfig()
        print()
        for model_name, embedding_dim, num_trf_blocks in name_embdim_numtrf:
            gpt_model_config.embedding_dim = embedding_dim
            gpt_model_config.num_trf_blocks = num_trf_blocks
            gpt_model_v1 = GPTModelV1()
            total_params = sum(p.numel() for p in gpt_model_v1.parameters())
            total_size_bytes = total_params * 4
            total_size_mb = total_size_bytes / (1024 * 1024)
            print(f"Total size of {model_name}: {total_size_mb:.2f} MB")

    def test_text_generation_without_training(self):
        torch.manual_seed(123)
        texts = (
            "In the heart of the city stood the old library, a relic from a bygone era.",
            "Its stone walls bore the marks of time, and ivy clung tightly to its facade.",
        )
        token_ids = TextProcessor.tokenize(
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
        max_new_tokens = 3
        GPTModelConfig().initizalize()
        gpt_model_v1 = GPTModelV1().eval()  # disable dropout (no training)
        for inputs, targets in data_iter:
            print("\nInputs:\n", inputs)
            print("Targets:\n", targets)
            for _ in range(max_new_tokens):
                inputs_cond = inputs[:, -Config().context_length :]
                with torch.no_grad():  # prediction without training
                    logits = gpt_model_v1(inputs_cond)[:, -1, :]
                    probas = torch.softmax(logits, dim=-1)
                    inputs_next = torch.argmax(probas, dim=-1, keepdim=True)
                    inputs = torch.cat((inputs, inputs_next), dim=1)
            inputs_outputs = inputs
            print("Inputs + Outputs:\n", inputs_outputs, "\n")
            for input_output in inputs_outputs:
                TextProcessor.decode(input_output, verbose=True)
