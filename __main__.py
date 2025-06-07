# Install Black Formatter using shift+alt+F as shortcut key

from config import Config
from text_splitter import TextSplitter


if __name__ == "__main__":

    words_list = TextSplitter().split(Config().text, verbose=True)
    # TODO: Embedding
