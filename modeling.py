"""
Modeling for getting language from readme words

"""
'''
*------------------*
|                  |
|     MODELING     |
|                  |
*------------------*
'''

##### IMPORTS #####


# data manipulation
import pandas as pd
import numpy as np

# nlp
import nltk
import nltk.sentiment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# models
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes as nb

# local
from lugo_explore import count_unique_words_by_language


##### FUNCTIONS #####


def analyze_unique_words(*args):
    """
    The `analyze_unique_words` function takes in multiple lists of words, generates n-grams for each
    list, finds common words across all lists, and returns a dictionary of unique words for each list.
    :return: The function `analyze_unique_words` returns a dictionary containing the unique words for
    each list of words provided as arguments. The keys of the dictionary correspond to the index of the
    list of words, and the values are dictionaries where the keys are the unique words and the values
    are the frequency of each unique word.
    """
    # Initialize dictionaries to store n-grams and unique words
    ngram_dicts = {}
    unique_word_dicts = {}
    # Generate n-grams for each list of words
    for i, words in enumerate(args):
        ngrams = pd.Series(nltk.ngrams(words, 1)).value_counts()
        ngram_dicts[i] = {f'{k[0]} ': v for k, v in ngrams.to_dict().items()}
    # Get sets of words for each language
    word_sets = {i: set(ngram_dict.keys()) for i, ngram_dict in ngram_dicts.items()}
    # Find common words
    common_words = set.intersection(*word_sets.values())
    # Find and store unique words for each list
    for i, word_set in word_sets.items():
        unique_words = word_set - common_words
        unique_word_dicts[i] = {key: ngram_dicts[i][key] for key in unique_words}
    # return unique
    return unique_word_dicts


def get_unique(train):
    """
    The function `get_unique` takes a training dataset as input, counts the unique words for each
    programming language, analyzes the unique words, and returns a list of all unique words across all
    languages.
    
    :param train: The `train` parameter is likely a dataset or a collection of data that is used for
    training a machine learning model. It is passed to the `get_unique` function to perform some
    operations on it
    :return: a list of unique words from the given input.
    """
    # get all words of each language
    javascript, java, objective_c, other, all_words = count_unique_words_by_language(train)
    # get list of dicts of each lang for unique words
    f = [list(analyze_unique_words(javascript, java, objective_c, other)[i].keys()) for i in range(4)]
    # make list of dicts into one list
    feat = f[0]
    feat.extend(f[1])
    feat.extend(f[2])
    feat.extend(f[3])
    # strip any spaces
    feat = [word.strip() for word in feat]
    return list(set(feat))


def get_words_by_language(train):
    """
    The function `get_words_by_language` takes a DataFrame `train` as input and returns lists of words
    used in each programming language, as well as a list of all words.
    
    :param train: The `train` parameter is a DataFrame that contains the training data. It likely has
    columns such as `top3other` which represents the top 3 programming languages for each row, and
    `lemmatized` which contains the lemmatized text data
    :return: a tuple containing five lists: javascript, java, objective_c, other, and all_words.
    """
    # Get the words used in each programming language
    javascript = [word for row in train[train.top3other=='JavaScript']['lemmatized'] for word in row.split()]
    java = [word for row in train[train.top3other=='Java']['lemmatized'] for word in row.split()]
    objective_c = [word for row in train[train.top3other=='Objective-C']['lemmatized'] for word in row.split()]
    other = [word for row in train[train.top3other=='other']['lemmatized'] for word in row.split()]
    all_words = [word for row in train['lemmatized'] for word in row.split()]
    return javascript, java, objective_c, other, all_words


def get_ngrams_by_language(javascript, java, objective_c, other,n):
    """
    The function `get_ngrams_by_language` takes in four sets of code snippets in different programming
    languages (JavaScript, Java, Objective-C, and other) and returns n-grams for each language.
    
    :param javascript: A string containing JavaScript code
    :param java: The `java` parameter is a string containing Java code
    :param objective_c: The parameter `objective_c` represents the code written in the Objective-C
    programming language
    :param other: The "other" parameter refers to a collection of code snippets or text that does not
    belong to any specific programming language. It could include code snippets from languages not
    mentioned in the function (e.g., Python, C++, etc.) or any other text that needs to be processed for
    n-grams
    :param n: The parameter "n" represents the number of words in each n-gram. It determines the size of
    the n-grams that will be generated for each language
    :return: four variables: `javascript`, `java`, `objective_c`, and `other`.
    """
    # make ngrams for each lang
    javascript = pd.Series(nltk.ngrams(javascript,n)).apply(lambda row: ' '.join([str(x) for x in row]))
    java = pd.Series(nltk.ngrams(java,n)).apply(lambda row: ' '.join([str(x) for x in row]))
    objective_c = pd.Series(nltk.ngrams(objective_c,n)).apply(lambda row: ' '.join([str(x) for x in row]))
    other = pd.Series(nltk.ngrams(other,n)).apply(lambda row: ' '.join([str(x) for x in row]))
    return javascript, java, objective_c, other


def get_unique2(train):
    """
    The function `get_unique2` takes a training dataset as input, extracts words from different
    programming languages, generates n-grams (unigrams, bigrams, and trigrams) for each language,
    concatenates them, analyzes unique words/n-grams for each language, and returns a list of unique
    features.
    
    :param train: The `train` parameter is likely a dataset or a collection of data that is used to
    train a model or perform some analysis. It is passed to the `get_unique2` function as an input
    :return: a list of unique features, which are words or n-grams (1, 2, or 3) that appear in the given
    train data.
    """
    # get all words of each language
    javascript, java, objective_c, other, all_words = get_words_by_language(train)
    # make a series from individual words by lang
    javascript_g, java_g, objective_c_g, other_g = get_ngrams_by_language(
        javascript, java, objective_c, other,1)
    # make new variables for them
    js = javascript_g
    j = java_g
    obj_c = objective_c_g
    o = other_g
    # make series for bigrams by lang
    javascript_g2, java_g2, objective_c_g2, other_g2 = get_ngrams_by_language(
        javascript, java, objective_c, other,2)
    # make series for trigrams by lang
    javascript_g3, java_g3, objective_c_g3, other_g3 = get_ngrams_by_language(
        javascript, java, objective_c, other,3)
    # concat them all together
    js = pd.concat([js,javascript_g2,javascript_g3])
    j = pd.concat([j,java_g2,java_g3])
    obj_c = pd.concat([obj_c,objective_c_g2,objective_c_g3])
    o = pd.concat([o,other_g2,other_g3])
    # get list of dicts of each lang for unique words/ngrams
    f1 = [list(analyze_unique_words(js, j, obj_c, o)[i].keys()) for i in range(4)]
    # make list of dicts into one list
    features = f1[0]
    features.extend(f1[1])
    features.extend(f1[2])
    features.extend(f1[3])
    # strip any spaces
    features = [word.strip() for word in features]
    # return list
    return list(set(features))


def make_cv(Xtr,Xv,Xt):
    """
    The function `make_cv` takes in three sets of data (train, validation, and test) and converts them
    into bag-of-words representations using a CountVectorizer with n-gram range of 1 to 3, and then
    returns the transformed data as dataframes.
    
    :param Xtr: Xtr is the training data, which is a pandas DataFrame containing the lemmatized text
    data
    :param Xv: Xv is the validation dataset, which is used to evaluate the performance of the model
    during training. It is a subset of the overall dataset that is not used for training the model but
    is used to tune the hyperparameters and assess the model's generalization ability
    :param Xt: Xt is the test data, which is a dataframe containing the text data that you want to
    classify or analyze
    :return: three dataframes: Xtr_cv, Xv_cv, and Xt_cv.
    """
    #make my bag of words up to trigrams cv and keep single characters
    cv = CountVectorizer(ngram_range=(1,3),token_pattern=r'(?u)\b\w+\b')
    # fit and transform train
    Xtr_bow_cv = cv.fit_transform(Xtr.lemmatized)
    # transform val and test
    Xv_bow_cv = cv.transform(Xv.lemmatized)
    Xt_bow_cv = cv.transform(Xt.lemmatized)
    # make dfs
    Xtr_cv = pd.DataFrame(Xtr_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xtr.index)
    Xv_cv = pd.DataFrame(Xv_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xv.index)
    Xt_cv = pd.DataFrame(Xt_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xt.index)
    return Xtr_cv,Xv_cv,Xt_cv


def make_tfidf(Xtr,Xv,Xt):
    """
    The function `make_tfidf` takes in three sets of data (train, validation, and test) and applies the
    TF-IDF vectorization technique to convert the text data into numerical features, using n-grams up to
    trigrams and keeping single characters. It then returns the transformed data as pandas DataFrames.
    
    :param Xtr: Xtr is the training data, which is a dataframe containing the text data that you want to
    transform into TF-IDF features. The "lemmatized" column in the dataframe contains the preprocessed
    text data
    :param Xv: Xv is the validation dataset, which is used to evaluate the performance of the model
    during training
    :param Xt: Xt is the input data for the test set. It is a dataframe containing the text data that
    needs to be transformed into TF-IDF representation
    :return: three dataframes: Xtr_tfidf, Xv_tfidf, and Xt_tfidf.
    """
    #make my bag of words up to trigrams tfidf and keep single characters
    tfidf = TfidfVectorizer(ngram_range=(1,3),token_pattern=r'(?u)\b\w+\b')
    # fit and transform train
    Xtr_bow_tfidf = tfidf.fit_transform(Xtr.lemmatized)
    # transform val and test
    Xv_bow_tfidf = tfidf.transform(Xv.lemmatized)
    Xt_bow_tfidf = tfidf.transform(Xt.lemmatized)
    # make dfs
    Xtr_tfidf = pd.DataFrame(Xtr_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xtr.index)
    Xv_tfidf = pd.DataFrame(Xv_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xv.index)
    Xt_tfidf = pd.DataFrame(Xt_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xt.index)
    return Xtr_tfidf,Xv_tfidf,Xt_tfidf


def class_models(Xtr,ytr,Xv,yv):
    """
    The function `class_models` trains and evaluates different classification models (Logistic
    Regression, Complement Naive Bayes, and Multinomial Naive Bayes) on the given training and
    validation data, and returns a dataframe containing the model names, parameters, training accuracy,
    and validation accuracy for each model.
    
    :param Xtr: The training data features (X) for the classification models
    :param ytr: The training labels for the classification model
    :param Xv: Xv is the feature matrix for the validation set. It contains the input features for each
    instance in the validation set
    :param yv: The parameter `yv` represents the target variable for the validation set. It is the true
    labels or classes for the validation set data
    :return: a pandas DataFrame containing the metrics for different models and their corresponding
    parameters. The metrics include the model name, parameters, training accuracy, and validation
    accuracy.
    """
    # baseline as mean
    pred_mean = ytr.value_counts(normalize=True)[0]
    output = {
            'model':'bl',
            'params':'None',
            'tr_acc':pred_mean,
            'v_acc':'?',
        }
    metrics = [output]
    # cycle through C,class_weight for log reg
    for c in [.01,.1,1,10,100,1000]:
        # logistic regression
        lr = LogisticRegression(C=c,class_weight='balanced',random_state=42,max_iter=500)
        lr.fit(Xtr,ytr)
        # accuracies
        ytr_acc = lr.score(Xtr,ytr)
        yv_acc = lr.score(Xv,yv)
        # table-ize
        output ={
                'model':'LogReg',
                'params':f"C={c},class_weight='balanced',max_iter=500",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    # cycle through alpha for CNB
    for a in np.arange(.1,.6,.1):
        # naive bayes complement
        cnb = nb.ComplementNB(alpha=a)
        cnb.fit(Xtr,ytr)
        # accuracies
        ytr_acc = cnb.score(Xtr,ytr)
        yv_acc = cnb.score(Xv,yv)
        # table-ize
        output ={
                'model':'CNB',
                'params':f'alpha={a}',
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    metrics_df = pd.DataFrame(metrics)
    # cycle through alpha for MNB
    for a in np.arange(.1,.6,.1):
        # naive bayes multinomial
        mnb = nb.MultinomialNB(alpha=a)
        mnb.fit(Xtr,ytr)
        # accuracies
        ytr_acc = mnb.score(Xtr,ytr)
        yv_acc = mnb.score(Xv,yv)
        # table-ize
        output ={
                'model':'MNB',
                'params':f'alpha={a}',
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


# ----------------------------------------------------------------------------
def get_Xs(train, validate, test):
    """
    The function `get_Xs` takes in three dataframes (train, validate, test) and returns the X and y
    values for each dataset.
    
    :param train: The training dataset, which contains the features and target variable for training the
    model
    :param validate: The `validate` parameter is a DataFrame that contains the validation dataset. It is
    used to evaluate the performance of the model during the training process and make adjustments if
    necessary
    :param test: The test parameter is a DataFrame containing the test data. It should have a column
    named 'lemmatized' which contains the lemmatized text data. The 'top3other' column should contain
    the target variable for the test data
    :return: six variables: X_train, X_val, X_test, y_train, y_val, and y_test.
    """
    # train
    X_train = train[['lemmatized']]
    y_train = train.top3other
    # validate
    X_val = validate[['lemmatized']]
    y_val = validate.top3other
    # test
    X_test = test[['lemmatized']]
    y_test = test.top3other
    return X_train, X_val, X_test, y_train, y_val, y_test


def cnb_model(Xtr,ytr,Xv,yv):
    """
    The function `cnb_model` trains a Complement Naive Bayes model using n-gram features and TF-IDF, and
    then prints the training and validation accuracies.
    
    :param Xtr: Xtr is the training data, which is a matrix or array-like object containing the features
    for training the model. Each row represents a sample, and each column represents a feature
    :param ytr: The parameter `ytr` represents the target variable (labels) for the training data `Xtr`.
    It is a 1-dimensional array or list containing the true labels for each sample in the training data
    :param Xv: Xv is the validation set input features. It is a matrix or array-like object that
    contains the input features for the validation set. Each row represents a sample, and each column
    represents a feature
    :param yv: The parameter "yv" represents the target variable for the validation set. It is a numpy
    array or pandas Series containing the true labels for the validation set
    """
    # ngram features and tfidf
    # naive bayes complement
    cnb = nb.ComplementNB(alpha=.5)
    cnb.fit(Xtr,ytr)
    # accuracies
    ytr_acc = cnb.score(Xtr,ytr)
    yv_acc = cnb.score(Xv,yv)
    # print results
    print('Complement Naive Bayes')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')


def test_model(Xtr,ytr,Xt,yt):
    """
    The function `test_model` trains a Complement Naive Bayes classifier on training data and evaluates
    its accuracy on test data.
    
    :param Xtr: The training data features (X) for the model
    :param ytr: The parameter `ytr` represents the target variable (labels) for the training data. It is
    a vector or array-like object containing the true labels for the training instances
    :param Xt: The variable `Xt` represents the test set features. It is a matrix or dataframe
    containing the input features for the test set. Each row in `Xt` corresponds to a single instance in
    the test set, and each column represents a different feature
    :param yt: yt is the true labels for the test set. It is the ground truth against which the
    predictions of the model will be compared
    """
    # ngram features and tfidf
    # naive bayes complement
    cnb = nb.ComplementNB(alpha=.5)
    cnb.fit(Xtr,ytr)
    # accuracies
    yt_acc = cnb.score(Xt,yt)
    # print results
    print('Complement Naive Bayes')
    print(f'Baseline Accuracy: {round(ytr.value_counts(normalize=True)[0],4)*100}%')
    print(f'Test Accuracy:     {round(yt_acc,4)*100}%')


def mnb_model(Xtr,ytr,Xv,yv):
    """
    The function `mnb_model` trains a Multinomial Naive Bayes model on the given training data and
    calculates the accuracy on both the training and validation sets.
    
    :param Xtr: The training set features. It is a matrix or array-like object of shape (n_samples,
    n_features), where n_samples is the number of samples in the training set and n_features is the
    number of features for each sample
    :param ytr: The parameter `ytr` represents the target variable (labels) for the training set. It is
    a one-dimensional array or list containing the true labels for each sample in the training set
    :param Xv: Xv is the validation set features. It is a matrix or array-like object that contains the
    input features for the validation set. Each row represents a sample, and each column represents a
    feature
    :param yv: The parameter `yv` represents the target variable for the validation set. It is the true
    labels or classes for the validation set observations
    """
    # all features and cv
    # naive bayes multinomial
    mnb = nb.MultinomialNB(alpha=.1)
    mnb.fit(Xtr,ytr)
    # accuracies
    ytr_acc = mnb.score(Xtr,ytr)
    yv_acc = mnb.score(Xv,yv)
    # print results
    print('Multinomial Naive Bayes')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')

def mnb_model2(Xtr,ytr,Xv,yv):
    """
    The function `mnb_model2` trains a Multinomial Naive Bayes model using n-gram features and
    cross-validation, and then prints the train and validation accuracies.
    
    :param Xtr: Xtr is the training data, which is a matrix or array-like object containing the features
    for training the model. Each row represents a sample, and each column represents a feature
    :param ytr: The parameter `ytr` represents the target variable (labels) for the training data. It is
    a one-dimensional array or list containing the true labels for each sample in the training set
    :param Xv: Xv is the validation set features. It is a matrix or array-like object that contains the
    input features for the validation set. Each row represents a sample, and each column represents a
    feature
    :param yv: The parameter "yv" in the function "mnb_model2" represents the target variable for the
    validation set. It is the true labels or classes for the validation data
    """
    # ngram features and cv
    # naive bayes multinomial
    mnb = nb.MultinomialNB(alpha=.5)
    mnb.fit(Xtr,ytr)
    # accuracies
    ytr_acc = mnb.score(Xtr,ytr)
    yv_acc = mnb.score(Xv,yv)
    # print results
    print('Multinomial Naive Bayes')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')


def log_model(Xtr,ytr,Xv,yv):
    """
    The function `log_model` trains a logistic regression model on the given training data and prints
    the train and validation accuracies.
    
    :param Xtr: The training set features (input variables) for the logistic regression model
    :param ytr: The parameter "ytr" represents the target variable (or labels) for the training set. It
    is a one-dimensional array or list containing the true labels for each sample in the training set
    :param Xv: The parameter Xv represents the features of the validation set. It is a matrix or
    array-like object that contains the input features for each sample in the validation set
    :param yv: The parameter "yv" represents the target variable for the validation set. It is the true
    labels or classes for the validation data
    """
    # unique features and tfidf
    # logistic regressor
    lr = LogisticRegression(C=1,class_weight='balanced',max_iter=500)
    lr.fit(Xtr,ytr)
    # accuracies
    ytr_acc = lr.score(Xtr,ytr)
    yv_acc = lr.score(Xv,yv)
    # print results
    print('Logistic Regression')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')










