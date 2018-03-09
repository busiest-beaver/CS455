import codecs
import os
import pickle
from os.path import basename, exists, join, splitext

import pandas as pd

class text_personality_classifier:
    def __init__(self):
        '''empty constructor'''

    def __get_model(self):
        models = {}
        traits = ["ope", "neu", "ext", "agr", "con"]
        
        for i in traits:
            if i == "con":
                i = "cons" #windows doesn't allow files names con

            file = open("/home/itadmin/src/CS455/text/pickles/"+i+".pkl",'rb')
            models[i] = pickle.load(file)

        return models

    def test(self, **kwargs):
        # column labels I want to use for LIWC
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
        
        traits = ["ope", "neu", "ext", "agr", "con"]  
        input_dir = kwargs['input_dir']
        models = self.__get_model()

        liwc_dir = ""
        if (os.path.isdir(input_dir+"/LIWC/")):
            liwc_dir = input_dir+"LIWC/"
        elif (os.path.isdir(input_dir+"LIWC/")):
            liwc_dir = input_dir+"/LIWC/"
        else:
            print("Test directory to LIWC not found.")
            exit()

        # Read in LIWC
        df = pd.read_csv(liwc_dir, error_bad_lines=False, engine="python")

        # predicting using LIWC
        X = df[LIWC_features]

        # inserting all the prediction in a dictionary
        for i in traits:
            clf = models.get(i)
            df[i] = clf.predict(X)

        results = {}
        for index, row in df.iterrows():
            traitsDict = {"ope": row["ope"],
                         "neu": row["neu"],
                         "ext": row["ext"],
                         "agr": row["agr"],
                         "con": row["con"]}
            userId = row["userId"]
            results[userId] = traitsDict

        return results
