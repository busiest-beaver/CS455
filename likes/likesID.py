import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import PassiveAggressiveRegressor
import pickle
import random



df1 = pd.read_csv('/Users/wildergarcia/Desktop/tcss455/training/profile/profile.csv', index_col=0)
df2 = pd.read_csv('/Users/wildergarcia/Desktop/tcss455/training/relation/relation.csv', index_col=0)

df = pd.merge(df1, df2, how="outer",on='userid') \
       .drop_duplicates().groupby('userid')['like_id'] \
       .apply(lambda x: ' '.join(x.astype(str))) \
       .reset_index()
# print (df)
# #sort the dataframe base in useid in profile and relation
df1.sort_values(['userid'], ascending=True)
# df4.sort_values(['userid'], ascending=True)
df.sort_values('userid', ascending=True)

#combine base in userid
df5 = pd.merge(df1, df, on=['userid'])
print(df5.head())


n = 8000
all_Ids = np.arange(len(df5))
random.shuffle(all_Ids)
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = df5.loc[test_Ids, :]
data_train = df5.loc[train_Ids, :]
# print(data_train)

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['like_id'])

y_train = data_train['gender']
clf = MultinomialNB() #declaring the type of machine learning
#training
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
newVec = CountVectorizer(vocabulary=count_vect.vocabulary_)
X_test = newVec.transform(data_test['like_id'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy Likes with naive-Bayes: %.2f" % accuracy_score(y_test,y_predicted))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("10-Fold Accuracy: %0.2f" % (scores.mean()))

with open("userlikes2.pkl", "wb") as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

with open("likeVectors2.pkl", "wb") as f:
    pickle.dump(newVec, f, pickle.HIGHEST_PROTOCOL)


