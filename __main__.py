# Install Black Formatter using shift+alt+F as shortcut key

from config import Config
from text_processor import TextProcessor


if __name__ == "__main__":

    text_processor = TextProcessor()
    words_list = text_processor.split(Config().text, verbose=True)

    # TODO: Embedding
