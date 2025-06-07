class TextSplitter:

    @staticmethod
    def split(text: str, verbose=False) -> list[str]:

        words_list = text.split()

        if verbose:
            word2idx_dict = {word: idx for idx, word in enumerate(words_list)}
            idx2word_dict = {idx: word for idx, word in enumerate(words_list)}
            idxs_list = [word2idx_dict[word] for word in words_list]

            print(
                """
list of words: {}
dictionary of word to index: {}    
dictionary of index to word: {}
list of indexes: {}
    """.format(
                    words_list, word2idx_dict, idx2word_dict, idxs_list
                )
            )

        return words_list
