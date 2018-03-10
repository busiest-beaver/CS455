import pickle
import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

#train model and export pickle binaries to be used in VM
def makePickle():
    random.shuffle(all_Ids)
    train_Ids = all_Ids[:] 
    data_train = df.loc[train_Ids, :]

    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(data_train['text'].values.astype('U'))
    newVec = CountVectorizer(vocabulary=count_vect.vocabulary_)
    y_train = data_train['gender']
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    with open("pickles/text_gender.pkl", "wb") as f:
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

    with open("pickles/text_vector.pkl", "wb") as f:
        pickle.dump(newVec, f, pickle.HIGHEST_PROTOCOL)

#get x-validation scores for a given ML technique
def getScores(clf):

    random.shuffle(all_Ids)
    train_Ids = all_Ids[:] 
    data_train = df.loc[train_Ids, :]

    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(data_train['text'].values.astype('U'))
    y_train = data_train['gender']
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    return scores.mean()

def makeScoreCSV():
    algorithms = ["MultinomialNB", "BernoulliNB"]#, "RandomForest"]

    algorithmUsed = []
    scores = []
    for i in range(len(algorithms)):
        for j in range(5):
            algorithmUsed.append(algorithms[i])
            score = 0
            if i == 0:
                score = getScores(MultinomialNB())
            elif i == 1:
                score = getScores(BernoulliNB())
            else:
                score = getScores(RandomForestClassifier())
            print(j+1, algorithms[i], score)
            scores.append(score)

    dataFrame = pd.DataFrame()
    dataFrame["algorithms"] = algorithmUsed
    dataFrame["scores"] = scores
    dataFrame.to_csv("scoresGender.csv")

df = pd.read_csv("processed.csv")
all_Ids = np.arange(len(df))
# makeScoreCSV()
makePickle()
