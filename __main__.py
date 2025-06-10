# Install Black Formatter using shift+alt+F as shortcut key

# pip install uv
# uv venv --python=python3.10 --seed
# source .venv/bin/activate
# uv pip install -r requirements.txt

from config import Config
from embedder import Embedder
from text_processor import TextProcessor


if __name__ == "__main__":

    text_processor = TextProcessor()
    token2id = text_processor.tokenize(
        Config().text, verbose=True, id_end=True, pair=True
    )

    embedder = Embedder()
    token_ids = [id for token, id in token2id]
    embedder.to_vector(token_ids)

    # TODO: Data sampling with a sliding window
