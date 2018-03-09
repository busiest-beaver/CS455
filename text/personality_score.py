import os
import random
from math import sqrt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,)
from sklearn.linear_model import (LinearRegression, Ridge, BayesianRidge)

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

def getScores(trait):
    all_Ids = np.arange(len(df))
    random.shuffle(all_Ids)
    test_Ids = all_Ids[0:n] 
    train_Ids = all_Ids[n:] 
    data_test = df.loc[test_Ids, :]
    data_train = df.loc[train_Ids, :]

    X_train = data_train[LIWC_features]
    y_train = data_train[trait]
    X_test = data_test[LIWC_features]
    y_test = data_test[trait]

    linreg = LinearRegression() #declaring the type of machine learning
    linreg.fit(X_train, y_train)
    y_linreg = linreg.predict(X_test)

    linreg_score = sqrt(metrics.mean_squared_error(y_test, y_linreg))
    print('LinearRegression: ', linreg_score)
    algorithms.append("LinearRegression")
    performance.append(linreg_score)    

    # ##########################################################################################################
    ada = AdaBoostRegressor()
    ada.fit(X_train, y_train)

    y_ada = ada.predict(X_test)

    ada_score = sqrt(metrics.mean_squared_error(y_test, y_ada))
    print("ADABoost: ", ada_score)
    algorithms.append("ADABoost")
    performance.append(ada_score) 
    # ##########################################################################################################
    boost = GradientBoostingRegressor()
    boost.fit(X_train, y_train)
    
    y_boost = boost.predict(X_test)

    boost_score = sqrt(metrics.mean_squared_error(y_test, y_boost))
    print("GradientBoost: ", boost_score)
    algorithms.append("GradientBoost")
    performance.append(boost_score) 
    # ##########################################################################################################
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    
    y_ridge = ridge.predict(X_test)

    ridge_score = sqrt(metrics.mean_squared_error(y_test, y_ridge))
    print("Ridge: ", ridge_score)
    algorithms.append("Ridge")
    performance.append(ridge_score) 
    # ##########################################################################################################
    bridge = BayesianRidge()
    bridge.fit(X_train, y_train)
    
    y_bridge = bridge.predict(X_test)

    bridge_score = sqrt(metrics.mean_squared_error(y_test, y_bridge))
    print("Bayesian Ridge: ", bridge_score)
    algorithms.append("Bayesian Ridge")
    performance.append(bridge_score) 
    # ##########################################################################################################
    print()
    ny = [0]*1500

    for i in range(len(ny)):
        ny[i] = (y_linreg[i] + y_boost[i] + y_ridge[i] + y_bridge[i] + y_ada[i])/5

    algorithms.append("MEAN")
    performance.append(sqrt(metrics.mean_squared_error(y_test, ny)))

df = pd.read_csv("processed.csv")
n = 1500
traits = ["ope", "neu", "ext", "agr", "con"]
for j in traits:
    algorithms = []
    performance = []
    for i in range(100):
        print(j.upper(), i+1)
        getScores(j)
    dfPerformance = pd.DataFrame()
    dfPerformance["algorithms"] = algorithms
    dfPerformance["scores"] = performance
    dfPerformance.to_csv('scores'+j.upper()+'.csv')
