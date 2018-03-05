import os
import pickle
from os.path import basename, exists, join, splitext
import pandas as pd

class liwc_age_classifier:

    def __init__(self):
        '''empty constructor'''

    def __get_model(self):
        file = open("/home/itadmin/src/CS455/text/liwc_age_classifier.pkl",'rb')
        model = pickle.load(file)
        return model

    def test(self, **kwargs):
        prediction = None
        input_dir = kwargs['input_dir']

        # loading the pickled NB model
        model = self.__get_model()

        if (model == None):
            return {}

        # checking if directory to the text files exist
        if (os.path.isdir(input_dir+"/LIWC/")):
            input_dir = input_dir+"/LIWC/"
        elif (os.path.isdir(input_dir+"LIWC/")):
            input_dir = input_dir+"/LIWC/"
        else:
            print("Test directory to LIWC not found.")
            exit()

        df = pd.read_csv(input_dir)
        #there is an inconsistency between 'userID' label in profile and LIWC, making sure the right column label is used
        if 'userid' in df.columns:
            df.rename(columns={'userid': 'userId'}, inplace=True)
        
        #column labels I want to use
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

        # using the count vector to predict gender
        prediction = model.predict(df[LIWC_features])

        # adding the gender column predicted by our model to the dataframe
        df['age'] = prediction

        # using the ID and age columns in our dataframe to create a dictionary
        results = dict(zip(df['userId'], df['age']))

        return results
