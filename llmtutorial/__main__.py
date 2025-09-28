# Install Black Formatter using shift+alt+F as shortcut key

# pip install uv
# uv venv --python=python3.10 --seed
# source .venv/bin/activate
# uv pip install -r requirements.txt

import os
import sys
import torch

from torch.utils.data import DataLoader
from .cli import CLI
from .config import Config

# TODO: implement gpt-model from Config #
from .gpt_model.embedder import Embedder
from .gpt_model.gpt_model_config import GPTModelConfig

#                                       #
from .text_processor import TextProcessor


class Main:

    @staticmethod
    def main():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, *Config().texts)
        texts = TextProcessor.file_to_text_data(file_path)
        print(
            "\n",
            "".join(
                TextProcessor.tokenize(texts, verbose=False, id_end=False, pair=False)
            ),
        )
        token2id = TextProcessor.tokenize(texts, verbose=True, id_end=True, pair=True)

        token_ids = [id for token, id in token2id]

        torch.manual_seed(Config().seed_num)

        Config().dataset = token_ids
        dataset = Config().dataset
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=Config().batch_size,
            shuffle=True,
            num_workers=Config().num_workers,
            drop_last=True,
        )
        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        print("\nInputs:\n", inputs)
        print("\nTargets:\n", targets)

        input_embeddings = Embedder.input_embeddings(inputs)
        print("\nSize of input embeddings:", input_embeddings.shape)

        context_vector_embeddings = GPTModelConfig().attention(input_embeddings)
        print("\nSize of context vector embeddings:", context_vector_embeddings.shape)


"""
python -m llmtutorial
python -m llmtutorial --help
python -m llmtutorial config
python -m llmtutorial config --textfile the-verdict.txt --cxtlen 256 --encoding gpt2
"""
if __name__ == "__main__":

    try:
        if len(sys.argv) > 1:
            CLI().app()
    finally:
        if len(sys.argv) == 1 or CLI().succeeded:
            Main.main()
