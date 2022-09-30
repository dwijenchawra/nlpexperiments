from threading import Thread
import numpy as np
import os
import re

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


import sys

ngram_min, ngram_max, min_df, max_df, c, penalty, filename = sys.argv[1:]


data_path = "/home/x-dchawra/nlpexperiments/rf_svm_imdb/IMDB Dataset.csv.zip"


data = pd.read_csv(data_path)
data.columns = ['text', 'label']
data['label'] = data.label.map({'negative': 0, 'positive': 1})

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punct_remover = str.maketrans('', '', punctuation)


def preprocess_text(text):
    text = re.sub('<.*?>', '', text)
    tokens = wordpunct_tokenize(text.lower())
    tokens = [i.translate(punct_remover) for i in tokens]
    tokens = [i for i in tokens if i not in stop_words]
    tokens = [wnl.lemmatize(i) for i in tokens]
    text = " ".join(tokens)

    return text


text_data = []
for i in tqdm(data.text):
    text_data.append(preprocess_text(i))
data['text'] = text_data

avg_tokens = sum([len(i.split()) for i in data.text]) / len(data.text)
print(f"Average Number of Tokens per Review: {round(avg_tokens)}")


def generate_features(data, test_size=0.2, feature_type='bow', ngram_min=1, ngram_max=1, min_df=1, max_df=1.0):
    """
    Function to generate features for train and test data

    Parameters
    ----------
    data - Dataframe with columns 'text' and 'label'
    test_size - Proportion of data that will be held out for evaluation
    feature_type - Type of features to use as represenation. Options are 'bow' or 'tfidf'
    max_df - When building the vocabulary ignore terms that have a document frequency strictly
             higher than the given threshold (corpus-specific stop words). If float, the parameter
             represents a proportion of documents, integer absolute counts. This parameter is ignored
             if vocabulary is not None.
    min_df - When building the vocabulary ignore terms that have a document frequency strictly lower than
            the given threshold. If float, the parameter represents a proportion of documents, integer
            absolute counts. This parameter is ignored if vocabulary is not None.

    Returns
    -------
    x_train - Input features for the train set
    y_train - Labels for the train set
    x_test - Input features for the test set
    y_test - Labels for the test set
    """

    # Split data into Train and Test
    train_df, test_df = train_test_split(data, test_size=0.2)

    # Instantiate Object to generate feature representations
    if feature_type == 'bow':
        vectorizer = CountVectorizer(ngram_range=(
            ngram_min, ngram_max), min_df=min_df, max_df=max_df)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(
            ngram_min, ngram_max), min_df=min_df, max_df=max_df)
    else:
        raise ValueError(
            "feature_type must be set to either 'bow' or 'tfidf'.")

    # Generate features for train and test set
    x_train = vectorizer.fit_transform(train_df.text)
    x_test = vectorizer.transform(test_df.text)
    y_train = train_df.label.values
    y_test = test_df.label.values

    return x_train, y_train, x_test, y_test







def optimizeModel(xtrain, xtest, train_labels, test_labels, c, penalty):
    # Define model parameters and train model
    svm_classifier = LinearSVC(penalty='l2', C=10, tol=1e-5, max_iter=5000)
    svm_classifier.fit(x_train, y_train)

    # Score Model
    svm_score = svm_classifier.score(x_test, y_test)
    test_pred = svm_classifier.predict(x_test)
    test_prec = precision_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)

    print("Evaluation Metrics")
    print('-' * 18)
    print(f"Precision: {test_prec}")
    print(f"Recall: {test_recall}")
    print(f"F1: {test_f1}")
    print()

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, test_pred, normalize='true')
    ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot()

    rf_classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    rf_classifier.fit(x_train, y_train)

    # Score Model and Generate confusion matrix
    svm_score = rf_classifier.score(x_test, y_test)
    test_pred = rf_classifier.predict(x_test)
    test_prec = precision_score(y_test, test_pred)
    test_recall = recall_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    print("Evaluation Metrics")
    print('-' * 18)
    print(f"Precision: {test_prec}")
    print(f"Recall: {test_recall}")
    print(f"F1: {test_f1}")
    print()

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, test_pred, normalize='true')
    ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot()





def modelTester(ngram_min, ngram_max, min_df, max_df, c, penalty):
    # feature_type = 'bow'
    # ngram_min = 1
    # ngram_max = 1
    # min_df = 40
    # max_df = 1.0

    x_train, y_train, x_test, y_test = generate_features(
        data, feature_type=feature_type, ngram_min=ngram_min, ngram_max=ngram_max, min_df=min_df, max_df=max_df
    )

    terms = vectorizer.get_feature_names()
    term_counts = x_train.toarray().sum(axis=0).tolist()
    term_counts = sorted(zip(terms, term_counts), key=lambda x: x[1])
    term_counts.reverse()
    num_terms = len(terms)
    least_common = term_counts[-10:]
    least_common.reverse()

    return optimizeModel(x_train, x_test, train_df.label.values, test_df.label.values, c=c, penalty=penalty)


best_acc = modelTester(int(ngram_min), int(ngram_max), np.double(min_df), np.double(max_df), float(c), str(penalty))

print("Params:" + filename + ":BestAcc:" + str(best_acc))
# job scheduler





