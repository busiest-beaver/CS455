import pickle

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

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
X = df[LIWC_features]
y = df["ageLabel"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

clf = RandomForestClassifier(criterion = 'gini')
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=15)
print("10-Fold Accuracy: %0.2f" % (scores.mean()))

with open("liwc_gender_classifier.pkl", "wb") as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
