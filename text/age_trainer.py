import pickle
import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("processedAge.csv")
n = 8000
all_Ids = np.arange(len(df))
random.shuffle(all_Ids)
test_Ids = all_Ids[0:n] 
train_Ids = all_Ids[n:] 
data_test = df.loc[test_Ids, :]
data_train = df.loc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['text'].values.astype('U'))
y_train = data_train['ageLabel']
newVec = CountVectorizer(vocabulary=count_vect.vocabulary_)
X_test = newVec.transform(data_test['text'].values.astype('U'))
y_test = data_test['ageLabel']

clf_svc = SVC(probability=True, decision_function_shape='ovr')#criterion="gini", max_depth=82) 
clf_svc.fit(X_train, y_train)
y_predicted_svm = clf_svc.predict(X_test)
# Reporting on classification performance
print("Accuracy Text with SVM: %.2f" % accuracy_score(y_test, y_predicted_svm))
scores = cross_val_score(clf_svc, X_train, y_train, cv=5)
print("10-Fold Accuracy: %0.2f" % (scores.mean()))

clf_rfc = RandomForestClassifier(criterion = 'gini')
clf_rfc.fit(X_train, y_train)
y_predicted_rfc = clf_rfc.predict(X_test)
# Reporting on classification performance
print("Accuracy Text with RFC: %.2f" % accuracy_score(y_test, y_predicted_rfc))
scores = cross_val_score(clf_rfc, X_train, y_train, cv=5)
print("10-Fold Accuracy: %0.2f" % (scores.mean()))

LIWC_features = ['Seg', 'WC', 'WPS', 'Sixltr', 'Dic', 'Numerals',
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

X_train = data_train[LIWC_features]
X_test = data_test[LIWC_features]
clf_rfc_liwc = RandomForestClassifier(criterion = 'gini')
clf_rfc_liwc.fit(X_train, y_train)
y_predicted_rfc_liwc = clf_rfc_liwc.predict(X_test)
scores = cross_val_score(clf_rfc_liwc, X_train, y_train, cv=15)
print("10-Fold Accuracy: %0.2f" % (scores.mean()))


y_predicted_final = [None]*8000
print(len(y_predicted_rfc), len(y_predicted_rfc_liwc), len(y_predicted_svm))
# ensemble
for i in range(len(y_predicted_final)):
    if (y_predicted_rfc[i] == y_predicted_rfc_liwc[i]):
        y_predicted_final[i] = y_predicted_rfc[i]
    elif (y_predicted_rfc[i] == y_predicted_svm[i]):
        y_predicted_final[i] = y_predicted_rfc[i]
    elif (y_predicted_rfc_liwc[i] == y_predicted_svm[i]):
        y_predicted_final[i] = y_predicted_rfc_liwc[i]
    else:
        print("crap")
        y_predicted_final[i] = "xx-24"

print("Accuracy Text with All 3: %.2f" % accuracy_score(y_test, y_predicted_final))

with open("clf_rfc.pkl", "wb") as f:
    pickle.dump(clf_rfc, f, pickle.HIGHEST_PROTOCOL)
with open("clf_rfc_liwc.pkl", "wb") as f:
    pickle.dump(clf_rfc_liwc, f, pickle.HIGHEST_PROTOCOL)
with open("clf_svm.pkl", "wb") as f:
    pickle.dump(clf_svc, f, pickle.HIGHEST_PROTOCOL)
with open("newvec.pkl", "wb") as f:
    pickle.dump(newVec, f, pickle.HIGHEST_PROTOCOL)
