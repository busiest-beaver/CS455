import os
import pickle
import random
from math import sqrt

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer  # machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 7
np.random.seed(seed)
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
X_test = data_test[LIWC_features]
y_test = data_test['neu']

# clf = LinearRegression() #declaring the type of machine learning
# #training
# clf.fit(pd.concat([X_train[:2000], X_train[:2000]]), pd.concat([y_train[:2000], y_train[:2000]]))
# y_predicted = clf.predict(X_test)

# # Reporting on classification performance
# print('RMSE LinReg:', sqrt(metrics.mean_squared_error(y_test, y_predicted)))
# ##########################################################################################################

# rfr = RandomForestRegressor()
# rfr.fit(pd.concat([X_train[2000:6500], X_train[2000:6500]]), pd.concat([y_train[2000:6500], y_train[2000:6500]]))

# y_pdsvr = rfr.predict(X_test)

# print('RMSE RFC:', sqrt(metrics.mean_squared_error(y_test, y_pdsvr)))
# ##########################################################################################################

# svr = SVR()
# svr.fit(pd.concat([X_train[3500:7500], X_train[500:1890]]), pd.concat([y_train[3500:7500], y_train[500:1890]]))

# y_pd = svr.predict(X_test)

# print('RMSE SVR:', sqrt(metrics.mean_squared_error(y_test, y_pd)))
##########################################################################################################
model = Sequential()

model.add(Dense(35, input_dim=82, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(20, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(35, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='normal'))

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.7, nesterov=False)

model.compile(sgd, loss='mse', metrics=['mse'])
# model.fit(pd.concat([X_train[6500:], X_train[6500:]]), pd.concat([y_train[6500:], y_train[6500:]]), epochs=30, verbose=False)
model.fit(X_train, y_train, epochs=50, verbose=False)

y_pred = model.predict(X_test)
print('RMSE with neural network:', sqrt(metrics.mean_squared_error(y_test, y_pred)))

# ny = [0]*1500
# for i in range(len(ny)):
#     ny[i] = (y_pred[i][0] + y_predicted[i] + y_pd[i] + y_pdsvr[i])/4

# print('RMSE with neural network:', sqrt(metrics.mean_squared_error(y_test, ny)))
