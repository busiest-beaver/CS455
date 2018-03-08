import os
import pickle
import random
from math import sqrt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_extraction.text import CountVectorizer  # machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
# from sklearn.svm import SVR

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

df = pd.read_csv("processed.csv")
n = 1500
def train(trait):
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
    #training
    linreg.fit(X_train, y_train)
    y_linreg = linreg.predict(X_test)

    # Reporting on classification performance
    linreg_score = sqrt(metrics.mean_squared_error(y_test, y_linreg))
    print('LINREG: ', linreg_score)
    algorithms.append("LINREG")
    performance.append(linreg_score)    
    # ##########################################################################################################

    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)

    y_forest = forest.predict(X_test)
    forest_score = sqrt(metrics.mean_squared_error(y_test, y_forest))
    print('FOREST: ', forest_score)
    algorithms.append("FOREST")
    performance.append(forest_score) 
    # ##########################################################################################################
    ada = AdaBoostRegressor()
    ada.fit(X_train, y_train)

    y_ada = ada.predict(X_test)

    ada_score = sqrt(metrics.mean_squared_error(y_test, y_ada))
    print("ADA: ", ada_score)
    algorithms.append("ADA")
    performance.append(ada_score) 
    # ##########################################################################################################
    bag = BaggingRegressor()
    bag.fit(X_train, y_train)
    
    y_bag = bag.predict(X_test)

    bag_score = sqrt(metrics.mean_squared_error(y_test, y_bag))
    print("BAG: ", bag_score)
    algorithms.append("BAG")
    performance.append(bag_score) 
    # ##########################################################################################################
    extra = ExtraTreesRegressor()
    extra.fit(X_train, y_train)
    
    y_extra = extra.predict(X_test)

    extra_score = sqrt(metrics.mean_squared_error(y_test, y_extra))
    print("EXTRA: ", extra_score)    
    algorithms.append("EXTRA")
    performance.append(extra_score) 
    # ##########################################################################################################
    boost = GradientBoostingRegressor()
    boost.fit(X_train, y_train)
    
    y_boost = boost.predict(X_test)

    boost_score = sqrt(metrics.mean_squared_error(y_test, y_boost))
    print("BOOST: ", boost_score)
    algorithms.append("BOOST")
    performance.append(boost_score) 
    # ##########################################################################################################

    print()
    ny = [0]*1500

    for i in range(len(ny)):
        ny[i] = (y_linreg[i] + y_forest[i] + y_ada[i] + y_bag[i] + y_extra[i] + y_boost[i])/6

    algorithms.append("MEAN")
    performance.append(sqrt(metrics.mean_squared_error(y_test, ny)))

traits = ["ope", "neu", "ext", "agr", "con"]

for j in traits:
    algorithms = []
    performance = []
    for i in range(50):
        print(j.upper(), i+1)
        train(j)
    dfPerformance = pd.DataFrame()
    dfPerformance["algorithms"] = algorithms
    dfPerformance["scores"] = performance
    dfPerformance.to_csv('scores'+j.upper()+'.csv')
