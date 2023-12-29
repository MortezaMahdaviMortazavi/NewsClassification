from hazm import Normalizer , WordTokenizer , Stemmer
from collections import Counter
from tqdm import tqdm
from dadmatools.models.normalizer import Normalizer

import pandas as pd
import numpy as np

"""
Miscellaneous          51308
Economy                20250
Politics               17803
Sport                  14328
Science and Culture    11959
Social                  8785
Literature and Art      5949
"""

MERGE_DICT = {
    "Economy":"Economy",
    "Sport":"Sport",
    "Miscellaneous":"Miscellaneous",
    "Science and Culture":"Science and Culture",
    "Literature and Art":"Literature and Art",
    "Social":"Social",
    "Politics":"Politics",
    "Economy.Industry":"Economy",
    "Miscellaneous.World News":"Miscellaneous",
    "Literature and Art.Art":"Literature and Art",
    "Miscellaneous.Happenings":"Miscellaneous",
    "Science and Culture.Science":"Science and Culture",
    "Politics.Iran Politics":"Politics",
    "Economy.Bank and Bourse":"Economy",
    "Miscellaneous.Islamic":"Miscellaneous",
    "Social.Religion":"Social",
    "Economy.Agriculture":"Economy",
    "Miscellaneous.Picture":"Miscellaneous",
    "Social.Women":"Social",
    "Economy.Commerce":"Economy",
    "Science and Culture.Science.Information and Communication Technology":"Science and Culture",
    "Science and Culture.Science.Book":"Science and Culture",
    "Sport.World Cup":"Sport",
    "Literature and Art.Art.Cinema":"Literature and Art",
    "Miscellaneous.Picture.Caricature":"Miscellaneous",
    "Literature and Art.Art.Music":"Literature and Art",
    "Economy.Dwelling and Construction":"Economy",
    "Literature and Art.Art.Theater":"Literature and Art",
    "Science and Culture.Science.Medicine and Remedy":"Science and Culture",
    "Literature and Art.Literature":"Literature and Art"
}


NUMBER_OF_EACH_CLASS = 5000

SAMPLE_PER_CLASS = {
    "Miscellaneous":51308 - NUMBER_OF_EACH_CLASS,
    "Economy":20250 - NUMBER_OF_EACH_CLASS,
    "Politics":17803 - NUMBER_OF_EACH_CLASS,
    "Sport":14328 - NUMBER_OF_EACH_CLASS,
    "Science and Culture":11959 - NUMBER_OF_EACH_CLASS,
    "Social":8785 - NUMBER_OF_EACH_CLASS,
    "Literature and Art":5949 - NUMBER_OF_EACH_CLASS,
}


def process_pipeline(df,undersampling_type='random'):
    assert undersampling_type in ['random','length']
    df = clean_dataframe(df)

    print("--- Undersampling started ---")
    if undersampling_type == 'random':
        df = undersample_randomly(df,'Category',SAMPLE_PER_CLASS)
    else:
        df = undersample_by_max_length(df,'Category','Text_Length',SAMPLE_PER_CLASS)

    print("--- Undersampling Finished ---")
    print("--- Cleaning and Normalizing the Texts is Started ---")
    tqdm.pandas()
    df['Text'] = df['Text'].progress_apply(cleantext_dadmatools)
    print("--- Cleaning and Normalizing Texts is Finished")
    print("New df length is",len(df))
    df = df.dropna()
    return df



def clean_dataframe(df):
    print("--- Merging classes started ---")
    df['Category'] = df['Category'].map(MERGE_DICT)
    print("--- Merging classes finished")
    print("--- Creating a column for displaying Length of text finished")
    df['Text_Length'] = df['Text'].apply(len)
    print("--- Creating a column for displaying Length of text finished")
    return df

def cleantext(text):
    normalizer = Normalizer(
        correct_spacing=True,
        remove_diacritics=True,
        remove_specials_chars=True,
        decrease_repeated_chars=True,
        persian_style=True,
        persian_numbers=True,
        unicodes_replacement=True,
        seperate_mi=False
    )
    
    normalized_text = normalizer.normalize(text)
    return normalized_text


def cleantext_dadmatools(text):
    normalizer = Normalizer(
        full_cleaning=False,
        unify_chars=True,
        refine_punc_spacing=True,
        remove_extra_space=True,
        remove_puncs=True,
        remove_html=True,
        remove_stop_word=False,
        replace_email_with="<EMAIL>",
        replace_number_with="<NUM>",
        replace_url_with="<URL>",
        replace_mobile_number_with='',
        replace_emoji_with='<EMOJI>',
        replace_home_number_with=''
    )
    return normalizer.normalize(text)

class Vocabulary:
    def __init__(self,texts):
        self.word2idx = {}
        self.idx2word = {}
        self.threshold = 5
        self.word_freq = Counter()
        self.build_vocab(texts)

    def __call__(self,text):
        tokenizer = WordTokenizer()
        tokens = tokenizer.tokenize(text)
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

    def build_vocab(self, texts):
        """
        Build the vocabulary based on a list of texts.

        Args:
        - texts: List of strings.
        """
        print("Start building the vocabulary")
        tokenizer = WordTokenizer()

        for text in tqdm(texts):
            tokens = tokenizer.tokenize(text)
            self.word_freq.update(tokens)

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

        idx = 2
        for word, freq in self.word_freq.items():
            if freq >= self.threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print("Finishing the creating process of building vocab")

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




def undersample_randomly(df, category_column, deletions_per_class):
    """
    Undersample a DataFrame based on a specified number of deletions for each class.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - category_column: str
        The column containing the class labels.
    - deletions_per_class: dict
        A dictionary specifying the number of deletions for each class.

    Returns:
    - undersampled_df: DataFrame
        The undersampled DataFrame.
    """

    if category_column not in df.columns:
        raise ValueError(f"Column '{category_column}' not found in the DataFrame.")

    # Create an empty list to store the undersampled data
    for class_label in df[category_column].unique():
        class_indices = df[df[category_column] == class_label].index
        deletions = deletions_per_class.get(class_label, 0)
        rows_to_delete = np.random.choice(class_indices, deletions, replace=False)
        df = df.drop(rows_to_delete)

    undersampled_df = df
    return undersampled_df


def undersample_by_max_length(df, category_column, text_length_column, deletions_per_class):
    """
    Undersample a DataFrame by deleting samples with the highest text lengths for each class.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - category_column: str
        The column containing the class labels.
    - text_length_column: str
        The column containing the text lengths.
    - deletions_per_class: dict
        A dictionary specifying the number of deletions for each class.

    Returns:
    - undersampled_df: DataFrame
        The undersampled DataFrame.
    """
    if category_column not in df.columns or text_length_column not in df.columns:
        raise ValueError(f"Columns '{category_column}' or '{text_length_column}' not found in the DataFrame.")
    
    for class_label in df[category_column].unique():
        class_indices = df[df[category_column] == class_label].index
        deletions = deletions_per_class.get(class_label, 0)
        sorted_indices = df.loc[class_indices].sort_values(by=text_length_column, ascending=False).index
        rows_to_delete = sorted_indices[:deletions]
        df = df.drop(rows_to_delete)

    return df



# if __name__ == "__main__":
#     df = pd.read_csv('dataset/train.csv',encoding='utf-8')
#     df = process_pipeline(df,undersampling_type='length')
#     df.to_csv('cleaned_train_length.csv')