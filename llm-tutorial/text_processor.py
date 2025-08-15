import tiktoken

from config import Config


class TextProcessor:

    @staticmethod
    def tokenize(texts: list[str], verbose=False, id_end=False, pair=False) -> list:

        bpe_tokenizer = tiktoken.get_encoding(Config().encoding)

        ids = bpe_tokenizer.encode(
            "<|endoftext|>".join(texts), allowed_special={"<|endoftext|>"}
        )
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
