
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
for i in tqdm(data.text):
    text_data.append(preprocess_text(i))
data['text'] = text_data

from ast import arg
import asyncio
import threading
import numpy as np
from threading import Thread


def modelTester(ngram_min, ngram_max, min_df, max_df):
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

    num_processors = -1

    model = LogisticRegression(
        max_iter=2000, n_jobs=num_processors, solver='liblinear')
    model.fit(x_train.toarray(), train_df.label.values)

    test_acc = model.score(x_test, test_df.label.values)
    print(f"Test Set Accuracy: {test_acc}")

    return test_acc


# threads = []
# for minfreq in np.arange(0, 0.1, 0.001):
#     ngram_min = 1
#     ngram_max = 4
#     min_df = minfreq
#     max_df = 1.0

#     t = ThreadWithReturnValue(target=modelTester, args=(
#         ngram_min, ngram_max, min_df, max_df))
#     t.start()

#     threads.append(t)

# results = []
# for i in threads: 
#     results.append(i.join())


print(modelTester(1, 4, 0.01, 1.0))

# currmax = 0.0
# curri = 0.0
# for i in np.arange(0.01, 0.001, -0.001):
#     print(i)
#     acc = modelTester(1, 4, i, 1.0)
#     if acc > currmax:
#         print("newmax")
#         print(acc)
#         print(i)
#         currmax = acc
#         curri = i

# print(currmax)
# print(curri)
