
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize
import os
import re
import pandas as pd
import nltk


import sys

ngram_min, ngram_max, min_df, max_df, c, penalty = sys.argv[1:]


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
data_path = "/home/x-dchawra/nlpexperiments/battelle_example/IMDB Dataset.csv.zip"


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

import numpy as np
from threading import Thread


def optimizeModel(xtrain, xtest, train_labels, test_labels, c, penalty):
    log_model = LogisticRegression(max_iter=2000, solver='liblinear', C=c, penalty=penalty)
    log_model.fit(xtrain, train_labels)

    best_test_acc = log_model.score(xtest, test_labels)
    print(f"Test Accuracy: {best_test_acc}")





def modelTester(ngram_min, ngram_max, min_df, max_df, c, penalty):
    train_df, test_df = train_test_split(data, test_size=0.2)
    vectorizer = CountVectorizer(ngram_range=(
        ngram_min, ngram_max), min_df=min_df, max_df=max_df)
    x_train = vectorizer.fit_transform(train_df.text)
    x_test = vectorizer.transform(test_df.text)

    terms = vectorizer.get_feature_names()
    term_counts = x_train.toarray().sum(axis=0).tolist()
    term_counts = sorted(zip(terms, term_counts), key=lambda x: x[1])
    term_counts.reverse()
    num_terms = len(terms)
    least_common = term_counts[-10:]
    least_common.reverse()

    optimizeModel(x_train, x_test, train_df.label.values, test_df.label.values, c=c, penalty=penalty)


modelTester(int(ngram_min), int(ngram_max), np.double(min_df), np.double(max_df), float(c), str(penalty))


# job scheduler






