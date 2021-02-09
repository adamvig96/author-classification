# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
from spacy import displacy
from collections import Counter
import hu_core_ud_lg

nlp = hu_core_ud_lg.load()


def lemmatizer(text):
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


# stopwords
swords = stopwords.words("hungarian") + ["ő", "ha"]


def text_process(tex, swords=swords):

    document = nlp(tex)

    tex = " ".join(
        [ent.text for ent in document if not ent.ent_type_]
    )  # remove named entities (should be revised)

    step_1 = lemmatizer(tex)  # lemmatization by spacy hungarian

    step_2 = [word.lower() for word in step_1 if not word.isdigit()]  # removing digits

    step_3 = "".join(
        [char for char in step_2 if char not in string.punctuation]
    )  # removing special characters

    step_4 = [
        word for word in step_3.split() if word.lower() not in swords
    ]  # removing stopwords

    text_return = " ".join(step_4)  # concating the text

    return text_return


# Read data

data_dir = "/Users/vigadam/Dropbox/My Mac (MacBook-Air.local)/Documents/Rajk/7. félév/Machine Learning/ML_project/author-classification/data/"

data_all = pd.read_pickle(data_dir + "all_site_2020.pkl")


# filter for 24.hu, drop None text
data = data_all.loc[data_all["page"] == "24.hu"].dropna(subset=["text"])

# run cleaner
data["processed_text"] = data["text"].apply(text_process)

data.to_pickle(data_dir + "24_cleaned.pkl")