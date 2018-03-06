import pickle
import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR

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

df = pd.read_csv("processedAge.csv")
n = 1500
all_Ids = np.arange(len(df))
random.shuffle(all_Ids)
test_Ids = all_Ids[0:n] 
train_Ids = all_Ids[n:] 
data_test = df.loc[test_Ids, :]
data_train = df.loc[train_Ids, :]
# print(data_train)

X_train = data_train[LIWC_features]
y_train = data_train['neu']
clf = LinearRegression() #declaring the type of machine learning
#training
clf.fit(X_train, y_train)

#ope,con,ext,agr,neu#
X_test = data_test[LIWC_features]
y_test = data_test['neu']
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print('MSE:', metrics.mean_squared_error(y_test, y_predicted))
