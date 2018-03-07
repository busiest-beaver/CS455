import random
import numpy as np
import pandas as pd #data processing
from sklearn.feature_extraction.text import CountVectorizer #machine learning
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("processedAge.csv")
n = 1500
all_Ids = np.arange(len(df))
random.shuffle(all_Ids)
test_Ids = all_Ids[0:n] 
train_Ids = all_Ids[n:] 
data_test = df.loc[test_Ids, :]
data_train = df.loc[train_Ids, :]
# print(data_train)

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['text'].values.astype('U'))
# print(X_train) #(user, word) frequency of word
y_train = data_train['ageLabel']
clf = MultinomialNB() #declaring the type of machine learning
#training
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
newVec = CountVectorizer(vocabulary=count_vect.vocabulary_)
X_test = newVec.transform(data_test['text'].values.astype('U'))
y_test = data_test['ageLabel']
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy Text with naive-Bayes: %.2f" % accuracy_score(y_test,y_predicted))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("10-Fold Accuracy: %0.2f" % (scores.mean()))

###########################################################
##########################################################

# liwc = list(df.columns[11:])
# liwcX = data_train[liwc]

# # clf = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy')
# clf = RandomForestClassifier(criterion = 'entropy') #max_depth=5, criterion='entropy')

# clf.fit(liwcX, y_train)

# liwcXtest = data_test[liwc]
# liwcPredicted = clf.predict(liwcXtest)
# scores = cross_val_score(clf, X_train, y_train, cv=10)
# print("10-Fold Accuracy: %0.2f" % (scores.mean()))
# print("Accuracy Text with LIWC-Tree: %.2f" % accuracy_score(y_test,liwcPredicted))


