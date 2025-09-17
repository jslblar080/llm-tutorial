# Install Black Formatter using shift+alt+F as shortcut key

# pip install uv
# uv venv --python=python3.10 --seed
# source .venv/bin/activate
# uv pip install -r requirements.txt

import torch

from torch.utils.data import DataLoader
from .config import Config

# TODO: implement gpt-model from Config #
from .gpt_model.embedder import Embedder
from .gpt_model.gpt_model_config import GPTModelConfig

#                                       #
from .text_processor import TextProcessor


# python -m llmtutorial
if __name__ == "__main__":

    texts = Config().texts

    text_processor = TextProcessor()
    print(
        "\n",
        "".join(
            text_processor.tokenize(texts, verbose=False, id_end=False, pair=False)
        ),
    )

    token2id = text_processor.tokenize(texts, verbose=True, id_end=True, pair=True)

    token_ids = [id for token, id in token2id]

    torch.manual_seed(123)

    Config().dataset = token_ids
    dataset = Config().dataset
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("\nInputs:\n", inputs)
    print("\nTargets:\n", targets)

    embedder = Embedder()
    input_embeddings = embedder.input_embeddings(inputs)
    print("\nSize of input embeddings:", input_embeddings.shape)

    GPTModelConfig().attention = input_embeddings
    attention = GPTModelConfig().attention
    context_vector_embeddings = attention(input_embeddings)
    print("\nSize of context vector embeddings:", context_vector_embeddings.shape)
