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

        # df = df1.merge(df2,on='userid').drop_duplicates()
        # df = df.sort_values(by='userid', ascending=True).groupby(['userid','gender'])
        # df = df.agg({'like_id':lambda x: ' '.join(x.astype(str))}).reset_index()


        df = pd.read_csv(input_dir+"relation.csv").astype(str).drop_duplicates()
        df = df.sort_values(by='userid', ascending=True).groupby('userid')

        df = df.agg({'like_id':lambda x:' '.join(x.astype(str))}).reset_index()
        like_ids = df['like_id']

        userids = df['userid']
        # userids = list(df.groups)

       
        # grouped = pd.read_csv(relation_path).astype(str).drop_duplicates().sort_values(by='userid', ascending=True).groupby('userid')
        # user_ids = list(grouped.groups)
        # like_ids = grouped.agg({'like_id':lambda x:' '.join(x)})['like_id']

        # predictions = dict(zip(userids, clf.predict(vec.transform(like_ids))))
        # return predictions


        vector = cv.transform(like_ids)

        # using the count vector to predict gender
        prediction = model.predict(vector)


        # using the ID and gender columns in our dataframe to create a dictionary
        results = dict(zip(userids, prediction))
        print(results)
        return results
