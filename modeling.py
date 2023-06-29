"""
Modeling for getting language from readme words

Functions:
- 
- 
- 
- 
"""


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
    javascript, java, objective_c, other, all_words = count_unique_words_by_language(train)
    f = [list(analyze_unique_words(javascript, java, objective_c, other)[i].keys()) for i in range(4)]
    feat = f[0]
    feat.extend(f[1])
    feat.extend(f[2])
    feat.extend(f[3])
    feat = [word.strip() for word in feat]
    return list(set(feat))


def make_cv(Xtr,Xv,Xt):
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


def cnb_model(Xtr,ytr,Xv,yv):
    # unique features and tfidf
    # naive bayes complement
    cnb = nb.ComplementNB(alpha=.5)
    cnb.fit(Xtr,ytr)
    # accuracies
    ytr_acc = cnb.score(Xtr,ytr)
    yv_acc = cnb.score(Xv,yv)
    print('Complement Naive Bayes')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')


def test_model(Xtr,ytr,Xt,yt):
    # unique features and tfidf
    # naive bayes complement
    cnb = nb.ComplementNB(alpha=.5)
    cnb.fit(Xtr,ytr)
    # accuracies
    yt_acc = cnb.score(Xt,yt)
    print('Complement Naive Bayes')
    print(f'Baseline Accuracy: {round(ytr.value_counts(normalize=True)[0],4)*100}%')
    print(f'Test Accuracy:     {round(yt_acc,4)*100}%')


def mnb_model(Xtr,ytr,Xv,yv):
    # all features and cv
    # naive bayes multinomial
    mnb = nb.MultinomialNB(alpha=.1)
    mnb.fit(Xtr,ytr)
    # accuracies
    ytr_acc = mnb.score(Xtr,ytr)
    yv_acc = mnb.score(Xv,yv)
    print('Multinomial Naive Bayes')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')


def log_model(Xtr,ytr,Xv,yv):
    # unique features and tfidf
    # logistic regressor
    lr = LogisticRegression(C=1,class_weight='balanced',max_iter=500)
    lr.fit(Xtr,ytr)
    # accuracies
    ytr_acc = lr.score(Xtr,ytr)
    yv_acc = lr.score(Xv,yv)
    print('Logistic Regression')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')










