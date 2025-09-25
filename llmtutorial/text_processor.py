import tiktoken

from torch import Tensor
from .config import Config


class TextProcessor:

    @staticmethod
    def file_to_text_data(file_path: str, verbose=False) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
            if verbose:
                print("Characters:", len(text_data))
            return text_data

    @staticmethod
    def tokenize(
        texts: tuple[str, ...] | str, verbose=False, id_end=False, pair=False
    ) -> list:

        encoding = Config().encoding

        bpe_tokenizer = tiktoken.get_encoding(encoding)

        if isinstance(texts, tuple):
            ids = bpe_tokenizer.encode(
                "<|endoftext|>".join(texts), allowed_special={"<|endoftext|>"}
            )
        if isinstance(texts, str):
            ids = bpe_tokenizer.encode(texts)
        bytes = [bpe_tokenizer.decode_single_token_bytes(id) for id in ids]
        tokens = [byte.decode("utf-8") for byte in bytes]
        token2id = list(zip(tokens, ids))
        id2token = list(zip(ids, tokens))

        category_idx = (id_end << 1) | pair
        category_dict = {
            0: ["token:", tokens],
            1: ["(id, token):", id2token],
            2: ["token id:", ids],
            3: ["(token, id):", token2id],
        }

        category_list = category_dict.get(category_idx, ["token:", tokens])

        if verbose:
            print("\n", category_list[0], category_list[1])

        return category_list[1]

    @staticmethod
    def decode(token_ids: Tensor, verbose=False) -> str:
        decoded_text = tiktoken.get_encoding(Config().encoding).decode(
            token_ids.tolist()
        )
        if verbose:
            print(decoded_text)
        return decoded_text
