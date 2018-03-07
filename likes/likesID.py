import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import random

from sklearn import tree
from sklearn import metrics
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import PassiveAggressiveRegressor



df1 = pd.read_csv('/Users/wildergarcia/Desktop/tcss455/training/profile/profile.csv', index_col=0)
df2 = pd.read_csv('/Users/wildergarcia/Desktop/tcss455/training/relation/relation.csv', index_col=0)

df = df1.merge(df2,on='userid').drop_duplicates().sort_values(by='userid', ascending=True).groupby(['userid','gender']).agg({'like_id':lambda x: ' '.join(x.astype(str))}).reset_index()

# Splitting the data into 300 training instances and 104 test instances
n = 1500
all_Ids = np.arange(len(df))
random.shuffle(all_Ids)
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test =df.loc[test_Ids, :]
data_train = df.loc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer() # this mean a transformation in the training data
X_train = count_vect.fit_transform(data_train['like_id']) # replace transcript with like_id
y_train = data_train['gender']
clf = MultinomialNB() # this is the place where you can decrare decision tree
clf.fit(X_train, data_train['gender'])

# Testing the Naive Bayes model
X_test = count_vect.transform(data_test['like_id'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("10-Fold Accuracy: %0.2f" % (scores.mean()))


with open("userlikes.pkl", "wb") as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

with open("likeVectors.pkl", "wb") as f:
    pickle.dump(count_vect, f, pickle.HIGHEST_PROTOCOL)
