class TextProcessor:

    @staticmethod
    def split(text: str, verbose=False, idx_value=False, dict=False):

        words_list = text.split()

        word2idx_dict = {word: idx for idx, word in enumerate(words_list)}
        idx2word_dict = {idx: word for idx, word in enumerate(words_list)}
        idxs_list = [word2idx_dict[word] for word in words_list]

        category_idx = (idx_value << 1) | dict
        category_dict = {
            0: ["list of words:", words_list],
            1: ["dictionary of index to word:", idx2word_dict],
            2: ["list of indexes:", idxs_list],
            3: ["dictionary of word to index:", word2idx_dict],
        }

        category_list = category_dict.get(category_idx, ["list of words: ", words_list])

        if verbose:
            print(category_list[0], category_list[1])

        return category_list[1]
