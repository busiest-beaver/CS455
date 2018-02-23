import codecs
import os
import pickle
from os.path import basename, exists, join, splitext
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class likes_gender_classifier:

    def __init__(self):
        '''empty constructor'''

    def __get_model(self):
        file = open("/home/itadmin/src/CS455/likes/userlikes.pkl",'rb')
        model = pickle.load(file)
        return model

    def __get_count_vectorizer(self):
        file = open("/home/itadmin/src/CS455/likes/likeVectors.pkl",'rb')
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
        if (os.path.isdir(input_dir+"/relation/")):
            input_dir = input_dir+"/relation/"
        elif (os.path.isdir(input_dir+"relation/")):
            input_dir = input_dir+"/relation"
        else:
            print("Test directory to statuses not found.")
            exit()

        df = pd.read_csv(input_dir+"relation.csv")

        df['like_id'] = df['like_id'].astype(str)
        list = []
        for i in df['like_id']:
            string = ' '.join(i)
            list.append(string)

        df['like_id'] = list #
        df = df.reset_index()

        vector = cv.transform(df['like_id'])

        # using the count vector to predict gender
        prediction = model.predict(vector)

        # adding the gender column predicted by our model to the dataframe
        df['gender'] = prediction

        # using the ID and gender columns in our dataframe to create a dictionary
        results = dict(zip(df['userid'], df['gender']))

        return results
