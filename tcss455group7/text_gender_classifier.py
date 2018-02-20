import codecs
import os
import pickle
from os.path import basename, exists, join, splitext
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class text_gender_classifier:

    def __init__(self):
        '''empty constructor'''

    def __get_model(self):
        file = open("/home/itadmin/src/CS455/text/text_model.pkl",'rb')
        model = pickle.load(file)
        return model

    def __get_count_vectorizer(self):
        file = open("/home/itadmin/src/CS455/text/text_count_vect.pkl",'rb')
        cv = pickle.load(file)
        return cv

    def test(self, **kwargs):
        prediction = None
        input_dir = kwargs['input_dir']

        # loading the pickled NB model
        model = self.__get_model()
        # loeading the count vectorizer created using test data to keep everything consistent
        cv = self.__get_count_vectorizer()

        if (model == None): 
            return {}

        # checking if directory to the text files exist
        if (os.path.isdir(input_dir+"/text/")):
            input_dir = input_dir+"/text/"
        elif (os.path.isdir(input_dir+"text/")):
            input_dir = input_dir+"/text"
        else:
            print("Test directory to statuses not found.")
            exit()

        # creating lists to store IDs and texts
        usersIDs = []
        texts = []
        df = pd.DataFrame()

        # populating the ID and text lists
        for fileName in os.listdir(input_dir):  
            fileContent = codecs.open(input_dir+fileName, "r", encoding="utf-8", errors="ignore").read()
            id = os.path.splitext(fileName)[0]
            usersIDs.append(id)
            texts.append(fileContent)

        # adding the ID and text lists to dataframe
        df['id'] = usersIDs
        df['text'] = texts

        # using dataframe to create count vector
        vector = cv.transform(df['text'].values.astype('U'))

        # using the count vector to predict gender
        prediction = model.predict(vector)

        # adding the gender column predicted by our model to the dataframe
        df['gender'] = prediction

        # using the ID and gender columns in our dataframe to create a dictionary
        results = dict(zip(df['id'], df['gender']))
        
        return results
