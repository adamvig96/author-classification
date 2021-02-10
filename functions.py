# Importing necessary libraries

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import string
import random
import time
import itertools
from tqdm import tqdm

tqdm.pandas()

import matplotlib.pyplot as plt
import seaborn as sns
import time

#%matplotlib inline

from PIL import Image
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
from spacy import displacy
from collections import Counter
#import hu_core_ud_lg

#nlp = hu_core_ud_lg.load()

from gensim import corpora
from gensim import models
import pyLDAvis
import pyLDAvis.gensim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score



def lemmatizer(text):
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)
#Additional stopwords

stopwords_manual = [
    "–",
    ":",
    "!",
    "-",
    ".",
    ",",
    ";",
    "+",
    "?",
    "/n",
    "–",
    "(",
    ")",
    "kb.",
    "ily",
    "oly",
    "kiemelt",
    "kép",
    "mondta",
    "ha",
    "két",
    "első",
    "korábban",
    "le",
    "közölte",
    "őket",
    "akár",
    "később",
    "előbb",
    "inkább",
    "előtt",
    "miatt",
    "például",
    "egyébként",
    "miután",
    "alapján",
    "végül",
    "ő",
    "három",
    "történt",
    "írta",
    "szó",
    "ugyanakkor",
    "szintén","korábbi","együtt","ám","során",
]

# 24.hu specific stopwords, basicly leftovers from html structure
stopwords_manual+=['gettyimages',"com", "amúgy", "pdt", "images","afp","getty" ]

swords = stopwords.words('hungarian') + stopwords_manual

def text_process_ours(tex, swords=swords):

    document = nlp(tex)
    tex = " ".join([ent.text for ent in document if not ent.ent_type_])
    step_1 = lemmatizer(tex)  # lemmatization by spacy hungarian
    step_2 = "".join(
        [char for char in step_1 if char not in string.punctuation]
    )  # removing special characters

    step_3 = [
        word for word in step_2.split() if word.lower() not in swords
    ]  # removing stopwords by

    step_4 = [word.lower() for word in step_3 if not word.isdigit()]  # removing digits

    text_return = " ".join(step_4)  # concating the text

    return text_return

def convert(list_element):
    tokens=list_element
    tokens = [token for token in tokens if len(token) > 3]
    return tokens

def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res



def build_model(vectorizer, estimators, x, y):
    pipelines = []

    if "Count Vectorizer" == vectorizer:
        #vec = CountVectorizer(analyzer=text_process_ours)
        vec = CountVectorizer()
    elif "Tfidf Vectorizer" == vectorizer:
        vec = TfidfVectorizer()
        #vec = TfidfVectorizer(analyzer=text_process_ours)

    vec.fit(x)

    for name in estimators:
        print("train %s" % name)

        if name == "Logistic Regression":
            estimator = LogisticRegression(solver="newton-cg", n_jobs=-1)
            pipeline = make_pipeline(vec, estimator)
        elif name == "One vs Rest":
            base_estimator = LogisticRegression(solver="newton-cg", n_jobs=-1)
            estimator = OneVsRestClassifier(base_estimator)
            pipeline = make_pipeline(vec, estimator)
        elif name == "Random Forest":
            estimator = RandomForestClassifier(n_jobs=-1)
            pipeline = make_pipeline(vec, estimator)
        elif name == "Support Vector Classifier":
            estimator = SVC(kernel="linear", gamma="auto",probability=True)
            pipeline = make_pipeline(vec, estimator)
        elif name == "Multinomial Naive Bayes":
            estimator = MultinomialNB(alpha=0.01)
            pipeline = make_pipeline(vec, estimator)
        elif name == "XGBoost Classifier":
            estimator = xgb.XGBClassifier(
                learning_rate=0.1,
                n_estimators=100,
                max_depth=10,
                subsample=0.8,
                colsample_bytree=1,
                gamma=1,
            )
            pipeline = make_pipeline(vec, estimator)

        pipeline.fit(x, y)
        pipelines.append({"name": name, "pipeline": pipeline})

    return pipelines, vec

def get_f_score(pipelines,X_train, y_train,X_test, y_test):
    out = {}
    for pipeline in pipelines:
        out[pipeline["name"]] = {
            "train": pipeline["pipeline"].score(X_train, y_train),
            "test": pipeline["pipeline"].score(X_test, y_test),
        }
    return pd.DataFrame(out).T

def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    #print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):

    start = time.time()

    def objective_function(params):
        NB_modell =  MultinomialNB(**params)
        score =f1_score(y_train, NB_modell.fit(X_train, y_train).predict(X_train), average="weighted")
        return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    best_param = fmin(
        objective_function,
        param_space,
        algo=tpe.suggest,
        max_evals=num_eval,
        trials=trials,
        rstate=np.random.RandomState(69),
    )

    loss = [x["result"]["loss"] for x in trials.trials]
    best_param_values = [x for x in best_param.values()]

    NB_modell_best = MultinomialNB(alpha=best_param_values[0])

    NB_modell_best.fit(X_train, y_train)

    print("")
    print("##### Results")
    print("Score best parameters: ", max(loss) * -1)
    print("Best parameters: ", best_param)
    print("Test Score: ", NB_modell_best.score(X_test, y_test))
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)

    return trials
