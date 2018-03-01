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
        # file = open("/data/userlikes.pkl",'rb')
        file = open("/home/itadmin/src/CS455/likes/userlikes.pkl",'rb')
        model = pickle.load(file)
        return model

    def __get_count_vectorizer(self):
        file = open("/home/itadmin/src/CS455/likes/likeVectors.pkl",'rb')
        # file = open("/data/likeVectors.pkl",'rb')
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

        # df = pd.read_csv(input_dir+"relation.csv")

        df = pd.read_csv(input_dir+"relation.csv").astype(str).drop_duplicates().groupby('userid')
        userids = list(df.groups)
        df = df.agg({'like_id':lambda x:' '.join(x)})#['like_id']
        like_ids = df['like_id']

        #print(like_ids.head())
        # df['like_id'] = df['like_id'].astype(str)
        # my_list = []
        # for i in df['like_id']:
        #     string = ' '.join(i)
        #     list.append(string)
        #
        # df['like_id'] = my_list #
        # df = df.reset_index()
        # print(df)

        vector = cv.transform(list(like_ids))

        # using the count vector to predict gender
        prediction = model.predict(vector)

        # adding the gender column predicted by our model to the dataframe
        #df['gender'] = prediction

        # using the ID and gender columns in our dataframe to create a dictionary
        results = dict(zip(userids, prediction))
        #print(results)
        return results
