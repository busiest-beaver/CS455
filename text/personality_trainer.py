import os
import pickle
import random
from math import sqrt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge

def makePickle(clf, trait):
    random.shuffle(all_Ids)
    train_Ids = all_Ids[:] 
    data_train = df.loc[train_Ids, :]

    X_train = data_train[LIWC_features]
    y_train = data_train[trait]

    clf.fit(X_train, y_train)
    if trait == "con":
        trait = "cons"

    with open(trait+".pkl", "wb") as f:
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

df = pd.read_csv("processed.csv")
all_Ids = np.arange(len(df))
LIWC_features = ['WC', 'WPS', 'Sixltr', 'Dic', 'Numerals',
       'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron',
       'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb',
       'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social',
       'family', 'friend', 'humans', 'affect', 'posemo', 'negemo', 'anx',
       'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat',
       'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel',
       'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion',
       'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig',
       'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon',
       'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth',
       'OtherP', 'AllPct']
traits = ["ope", "neu", "ext", "agr", "con"]

for i in traits:
    if i == "neu":
        makePickle(BayesianRidge(), i)
    else:
        makePickle(GradientBoostingRegressor(), i)
