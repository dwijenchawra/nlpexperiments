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

'''
tell people about what the confusion matrix means
what parts are important to us
'''

import sys

ngram_min, ngram_max, min_df, max_df, featuretype, filename = sys.argv[1:]


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

    return x_train, y_train, x_test, y_test, vectorizer







def optimizeSVMModel(xtrain, xtest, ytrain, ytest, c, penalty):
    # Define model parameters and train model
    grid = GridSearchCV(LinearSVC(penalty='l2', C=10, tol=1e-5,
                        max_iter=5000), parameters, n_jobs=-1)
    grid.fit(xtrain, ytrain)

    best_model = grid.best_estimator_
    best_test_acc = best_model.score(xtest, ytest)

    # Score Model and Generate confusion matrix
    test_pred = best_model.predict(xtest)
    test_prec = precision_score(ytest, test_pred)
    test_recall = recall_score(ytest, test_pred)
    test_f1 = f1_score(ytest, test_pred)

    # Generate confusion matrix
    # conf_matrix = confusion_matrix(ytest, test_pred, normalize='true')
    # ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot()

    return best_test_acc, test_prec, test_recall, test_f1


def optimizeRFModel(xtrain, xtest, ytrain, ytest):
    n_estimators = [int(x) for x in np.arange(start=50, stop=3000, step=100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.arange(1, 20, step=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 8, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 8, 16]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores

    grid = GridSearchCV(RandomForestClassifier(), param_grid=random_grid, n_jobs=-1)
    grid.fit(xtrain, ytrain)

    best_model = grid.best_estimator_
    best_test_acc = best_model.score(xtest, ytest)

    # Score Model and Generate confusion matrix
    test_pred = best_model.predict(xtest)
    test_prec = precision_score(ytest, test_pred)
    test_recall = recall_score(ytest, test_pred)
    test_f1 = f1_score(ytest, test_pred)

    return best_test_acc, test_prec, test_recall, test_f1
    


def svmModelTester(ngram_min, ngram_max, min_df, max_df, c, penalty, feature_type = 'bow'):
    # feature_type = 'bow'
    # ngram_min = 1
    # ngram_max = 1
    # min_df = 40
    # max_df = 1.0



    x_train, y_train, x_test, y_test = generate_features(
        data, feature_type=feature_type, ngram_min=ngram_min, ngram_max=ngram_max, min_df=min_df, max_df=max_df
    )

    # terms = vectorizer.get_feature_names()
    # term_counts = x_train.toarray().sum(axis=0).tolist()
    # term_counts = sorted(zip(terms, term_counts), key=lambda x: x[1])
    # term_counts.reverse()
    # num_terms = len(terms)
    # least_common = term_counts[-10:]
    # least_common.reverse()

    return optimizeSVMModel(x_train, x_test, y_train, y_test, c=c, penalty=penalty)


def rfModelTester(ngram_min, ngram_max, min_df, max_df, feature_type = 'bow'):
    # feature_type = 'bow'
    # ngram_min = 1
    # ngram_max = 1
    # min_df = 40
    # max_df = 1.0

    x_train, y_train, x_test, y_test = generate_features(
        data, feature_type=feature_type, ngram_min=ngram_min, ngram_max=ngram_max, min_df=min_df, max_df=max_df
    )

    # terms = vectorizer.get_feature_names()
    # term_counts = x_train.toarray().sum(axis=0).tolist()
    # term_counts = sorted(zip(terms, term_counts), key=lambda x: x[1])
    # term_counts.reverse()
    # num_terms = len(terms)
    # least_common = term_counts[-10:]
    # least_common.reverse()


    return optimizeRFModel(
        x_train, x_test, y_train, y_test)


# print("Params:" + filename + ":BestAcc:" + str(best_acc))
# job scheduler
rfModelTester(ngram_min, ngram_max, min_df, max_df, featuretype)




