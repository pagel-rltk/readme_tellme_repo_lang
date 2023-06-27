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
    # return prepped df
    return df






