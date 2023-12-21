from hazm import Normalizer , WordTokenizer , Stemmer
from collections import Counter


def cleantext(text):
    normalizer = Normalizer(
        correct_spacing=True,
        remove_diacritics=True,
        remove_specials_chars=True,
        decrease_repeated_chars=True,
        persian_style=True,
        persian_numbers=True,
        unicodes_replacement=False,
        seperate_mi=False
    )
    
    normalized_text = normalizer.normalize(text)
    return normalized_text

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

    def build_vocab(self, texts):
        """
        Build the vocabulary based on a list of texts.

        Args:
        - texts: List of strings.
        """
        tokenizer = WordTokenizer()

        for text in texts:
            tokens = tokenizer.tokenize(text)
            self.word_freq.update(tokens)

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

        idx = 2
        for word, freq in self.word_freq.items():
            if freq >= 2:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def word_to_index(self, word):
        """
        Convert a word to its corresponding index in the vocabulary.

        Args:
        - word: The word to convert.

        Returns:
        - The index of the word. Returns 1 if the word is not in the vocabulary.
        """
        return self.word2idx.get(word, 1)

    def index_to_word(self, index):
        """
        Convert an index to its corresponding word in the vocabulary.

        Args:
        - index: The index to convert.

        Returns:
        - The word corresponding to the index. Returns '<UNK>' if the index is not in the vocabulary.
        """
        return self.idx2word.get(index, '<UNK>')

    def tokenize(self, text):
        """
        Tokenize a text and convert it to a list of indices.

        Args:
        - text: The text to tokenize.

        Returns:
        - A list of indices corresponding to the tokens in the text.
        """
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize(text)
        return [self.word_to_index(token) for token in tokens]

    def get_vocab_size(self):
        """
        Get the size of the vocabulary.

        Returns:
        - The number of unique words in the vocabulary.
        """
        return len(self.word2idx)

