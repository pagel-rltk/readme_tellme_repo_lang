"""
Prepare GitHub Readme data

Functions:
- basic_clean
- tokenize
- stem
- lemmatize
- remove_stopwords
- prepare_readmes
"""

##### IMPORTS #####

import numpy as np
import pandas as pd
import unicodedata
import re
import nltk 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

##### FUNCTIONS #####

def basic_clean(words):
    """
    The function takes a string of words, converts them to lowercase, normalizes characters, removes
    unwanted characters, and replaces certain characters with spaces.
    
    :param words: The input string that needs to be cleaned. It can contain letters, numbers,
    punctuation marks, and special characters
    :return: The function `basic_clean` returns a cleaned version of the input `words` string, where all
    characters are converted to lowercase, non-ASCII characters are removed, and certain special
    characters like punctuation marks are removed or replaced with spaces.
    """
    # lowercase all
    words_low = words.lower()
    # normalize characters
    words_uni = unicodedata.normalize('NFKD',words_low).encode('ascii','ignore').decode('utf-8')
    # drop characters like -./,!?
    words_re = re.sub(r'[^a-z0-9\'\-\.\/\[\]\{\}\s]','',words_uni)
    return re.sub(r'[\-\.\/\[\]\{\}]',' ',words_re)

def tokenize(words_bc):
    """
    The function takes a string of words and uses the ToktokTokenizer from the NLTK library to tokenize
    the words and return them as a string.
    
    :param words_bc: The input string that needs to be tokenized
    :return: The function `tokenize` returns a string of tokenized words. The input `words_bc` is
    tokenized using the `ToktokTokenizer` from the `nltk.tokenize` module and the resulting tokens are
    returned as a string.
    """
    # make tokenizer
    tok = nltk.tokenize.ToktokTokenizer()
    # tokenize them
    return tok.tokenize(words_bc,return_str=True)

def stem(words_tok):
    """
    The function takes in a string of words, applies the Porter stemming algorithm to each word, and
    returns the stemmed words as a single string.
    
    :param words_tok: The input to the function is a string of words that have already been tokenized
    (split into individual words)
    :return: The function `stem` takes a string of words as input, tokenizes the words, applies Porter
    stemming algorithm to each word, and returns a string of stemmed words joined by spaces.
    """
    # make stemmer
    p_stem = nltk.PorterStemmer()
    # stem them
    return ' '.join([p_stem.stem(word) for word in words_tok.split()])

def lemmatize(words_tok):
    """
    The function lemmatizes a given string of words using the WordNetLemmatizer from the Natural
    Language Toolkit (nltk) and returns the lemmatized string.
    
    :param words_tok: The parameter "words_tok" is expected to be a string of words separated by spaces.
    The function will split this string into individual words, lemmatize each word using the
    WordNetLemmatizer from the NLTK library, and then join the lemmatized words back into a
    :return: The function `lemmatize` takes a string of words as input, tokenizes it, lemmatizes each
    word using WordNetLemmatizer, and returns a string of lemmatized words joined by spaces.
    """
    # make lemmatizer
    lem = nltk.stem.WordNetLemmatizer()
    # lemmatize them
    return ' '.join([lem.lemmatize(word) for word in words_tok.split()])

def remove_stopwords(words,extra_words=None,exclude_words=None):
    """
    This function removes stopwords from a given string of words, with the option to add or exclude
    additional stopwords.
    
    :param words: a string of words that you want to remove stopwords from
    :param extra_words: A list of additional words to be added to the default list of stopwords. These
    are words that are commonly used but do not carry much meaning in the context of text analysis
    :param exclude_words: A list of words that should be excluded from the stopwords list. These words
    will not be removed from the input text
    :return: a string that contains the words from the input string `words` with the stopwords removed.
    The stopwords are defined as common English words that do not carry much meaning, such as "the",
    "and", "a", etc. The function also allows for additional words to be added to the stopwords list
    (`extra_words`) or for specific words to be excluded from the stopwords list (`
    """
    # make stopwords list
    stop = stopwords.words('english')
    # add more stopwords
    if extra_words is not None:
        stop = list(set(stop).union(set(extra_words)))
    # remove stopwords
    if exclude_words is not None:
        stop = list(set(stop) - set(exclude_words))
    # return filtered with no stopwords
    return ' '.join([word for word in words.split() if word not in stop])


def prep_readmes(df):
    """
    This function prepares a dataframe by filtering out null values, cleaning and lemmatizing the
    'readme_contents' column, and creating new columns for the cleaned and lemmatized data.
    
    :param df: The input parameter is a pandas DataFrame containing information about GitHub
    repositories, including the contents of their README files and the programming language used
    :return: a preprocessed dataframe with the following columns: 'repo', 'language', 'readme_contents',
    'clean', and 'lemmatized'. The 'clean' column is derived from the 'readme_contents' column by
    removing stopwords and applying basic cleaning techniques. The 'lemmatized' column is derived from
    the 'clean' column by lemmatizing the text.
    """
    # Filter out nulls: 'readme_contents'
    df = df[df['readme_contents'] != ""]
    # Filter out nulls: 'language'
    df = df[df['language'].notna()]
    # reset index after null removal
    df = df.reset_index().drop(columns='index')
    # Derive column 'clean' from column: cleanup up 'readme_contents'
    df = df.assign(clean = df.apply(lambda row : remove_stopwords(tokenize(basic_clean(row.readme_contents)),"'"), axis=1))
    # Derive column 'lemmatized' from column: lemmatized 'clean'
    df = df.assign(lemmatized = df.apply(lambda row : lemmatize(row.clean), axis=1))
    # add column for top3 and other languages to make predictions simpler
    df['top3other'] = np.where(df.language == 'JavaScript','JavaScript','other')
    df['top3other'] = np.where(df.language == 'Objective-C','Objective-C',df['top3other'])
    df['top3other'] = np.where(df.language == 'Java','Java',df['top3other'])
    # return prepped df
    return df


def split_data(df, strat, seed=42, test=.2, validate=.25):
    """
    The function `split_data` takes in a dataframe `df`, a stratification variable `strat`, and optional
    arguments for the random seed, test size, and validation size, and returns three separate dataframes
    for training, validation, and testing.
    
    :param df: The input dataframe that you want to split into train, validation, and test sets
    :param strat: The strat parameter is used for stratified sampling. It is a column name in the
    dataframe df that is used to group the data before splitting. This is useful when you want to ensure
    that the train, validation, and test sets have similar distributions of a specific variable
    :param seed: The seed parameter is used to ensure reproducibility of the random splitting of the
    data. By setting a specific seed value, the same random splits will be generated each time the
    function is called with the same dataset, defaults to 42 (optional)
    :param test: The "test" parameter represents the proportion of the data that should be allocated for
    testing. In this case, it is set to 0.2, which means that 20% of the data will be used for testing
    :param validate: The `validate` parameter represents the proportion of the data that will be used
    for validation. It is a decimal value between 0 and 1, where 0 represents no validation data and 1
    represents all data used for validation
    :return: three dataframes: train, validate, and test.
    """
    # prep strat variable
    st = [strat]
    # split train_val and test
    train_validate, test = train_test_split(df, test_size=test, random_state=seed, stratify=df[st])
    # split train and val
    train, validate = train_test_split(train_validate, 
                                        test_size=validate, 
                                        random_state=seed, 
                                        stratify=train_validate[st])
    # return train, val and test
    return train, validate, test




