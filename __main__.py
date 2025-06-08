# Install Black Formatter using shift+alt+F as shortcut key

# pip install uv
# uv venv --python=python3.10
# source .venv/bin/activate
# uv pip install -r requirements.txt

from config import Config
from embedder import Embedder
from text_processor import TextProcessor


if __name__ == "__main__":

    text_processor = TextProcessor()
    word2idx_dict = text_processor.split(
        Config().text, verbose=True, idx_value=True, dict=True
    )

    embedder = Embedder()
    embedder.to_vector(word2idx_dict)

    # TODO: Specify text splitting and tokenizing
