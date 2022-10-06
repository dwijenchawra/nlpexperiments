import numpy as np
import re

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from prompt_toolkit import print_formatted_text
from pytest import param
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

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
for i in data.text:
    text_data.append(preprocess_text(i))
data['text'] = text_data

# avg_tokens = sum([len(i.split()) for i in data.text]) / len(data.text)
# print(f"Average Number of Tokens per Review: {round(avg_tokens)}")


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


def optimizeSVMModel(ngram_min, ngram_max, min_df, max_df, feature_type='bow'):
    xtrain, ytrain, xtest, ytest = generate_features(
        data, feature_type=feature_type, ngram_min=ngram_min, ngram_max=ngram_max, min_df=min_df, max_df=max_df
    )

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


def optimizeRFModel(ngram_min, ngram_max, min_df, max_df, feature_type='bow'):

    xtrain, ytrain, xtest, ytest = generate_features(
        data, feature_type=feature_type, ngram_min=ngram_min, ngram_max=ngram_max, min_df=min_df, max_df=max_df
    )
    # n_estimators = [int(x) for x in np.arange(start=50, stop=3000, step=100)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.arange(1, 20, step=5)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 4, 8, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [2, 4, 8, 16]

    # testing cv
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)] # 2nd run: 50-1000?
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 50, num=5)] # 2nd run: max < 110
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    grid = RandomizedSearchCV(RandomForestClassifier(), param_distributions=random_grid, n_jobs=-1, n_iter=100, random_state=42, verbose=0, pre_dispatch=16)
    grid.fit(xtrain, ytrain)

    # print(grid.cv_results_)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_test_acc = best_model.score(xtest, ytest)

    # Score Model and Generate confusion matrix
    test_pred = best_model.predict(xtest)
    test_prec = precision_score(ytest, test_pred)
    test_recall = recall_score(ytest, test_pred)
    test_f1 = f1_score(ytest, test_pred)

    return best_test_acc, test_prec, test_recall, test_f1, best_params
    

# job scheduler
best_test_acc, test_prec, test_recall, test_f1, params = optimizeRFModel(int(ngram_min), int(ngram_max), np.double(min_df), np.double(max_df), str(featuretype))

print(f"NLPParams;{filename};RFParams;{params};BestTestAccuracy;{best_test_acc};Precision;{test_prec};Recall;{test_recall};F1;{test_f1}")


