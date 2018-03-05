import codecs
import os
import pickle
from os.path import basename, exists, join, splitext

import pandas as pd

class text_age_classifier:
    def __init__(self):
        '''empty constructor'''

    def __get_model(self):
#         svm_file = open("/home/itadmin/src/CS455/text/clf_svm.pkl",'rb')
        rfc_file = open("/home/itadmin/src/CS455/text/clf_rfc.pkl",'rb')
#         rfc_liwc_file = open("/home/itadmin/src/CS455/text/clf_rfc_liwc.pkl",'rb')
        
#         svm_model = pickle.load(svm_file)
        rfc_model = pickle.load(rfc_file)
#         rfc_liwc_model = pickle.load(rfc_liwc_file)
        
        return rfc_model #, rfc_liwc_model, svm_model

    def __get_count_vectorizer(self):
        file = open("/home/itadmin/src/CS455/text/newvec.pkl",'rb')
        cv = pickle.load(file)
        return cv

    def test(self, **kwargs):
        prediction = None
        input_dir = kwargs['input_dir']

        # loading the pickled NB model
#         rfc_model, rfc_model_liwc, svm_model = self.__get_model()
        rfc_model = self.__get_model()

        if (rfc_model == None): ##or rfc_model_liwc == None or svm_model == None):
            return {}

#         liwc_dir = ""
#         # checking if directory to the text files exist
#         if (os.path.isdir(input_dir+"/LIWC/")):
#             liwc_dir = input_dir+"/LIWC/"
#         elif (os.path.isdir(input_dir+"LIWC/")):
#             liwc_dir = input_dir+"/LIWC/"
#         else:
#             print("Test directory to LIWC not found.")
#             exit()

        text_dir = ""
        # checking if directory to the text files exist
        if (os.path.isdir(input_dir+"/text/")):
            text_dir = input_dir+"/text/"
        elif (os.path.isdir(input_dir+"text/")):
            text_dir = input_dir+"/text"
        else:
            print("Test directory to statuses not found.")
            exit()

#         # Read in LIWC
#         df = pd.read_csv(liwc_dir, error_bad_lines=False)
#         #there is an inconsistency between 'userID' label in profile and LIWC, making sure the right column label is used
#         if 'userid' in df.columns:
#             df.rename(columns={'userid': 'userId'}, inplace=True)

        # Read in text files
        usersIDs = []
        texts = []
        df_text = pd.DataFrame()
        
        # populating the ID and text lists
        for fileName in os.listdir(text_dir):
            fileContent = codecs.open(text_dir+fileName, "r", encoding="utf-8", errors="ignore").read()
            id = os.path.splitext(fileName)[0]
            usersIDs.append(id)
            texts.append(fileContent)

        # adding the ID and text lists to dataframe
        df_text['userId'] = usersIDs
        df_text['text'] = texts

#         df.merge(df_text, 'outer', 'userId')
        
        #count vector for texts
        cv = self.__get_count_vectorizer()
        X_test = cv.transform(df_text['text'].values.astype('U'))

#         y_predicted_svm = svm_model.predict(X_test)

        y_predicted_rfc = rfc_model.predict(X_test)

#         #column labels I want to use for LIWC
#         LIWC_features = ['Seg', 'WC', 'WPS', 'Sixltr', 'Dic', 'Numerals',
#        'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron',
#        'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb',
#        'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social',
#        'family', 'friend', 'humans', 'affect', 'posemo', 'negemo', 'anx',
#        'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat',
#        'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel',
#        'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion',
#        'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig',
#        'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon',
#        'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth',
#        'OtherP', 'AllPct']

#         #predicting using LIWC
#         X_test = df[LIWC_features]
#         y_predicted_rfc_liwc = rfc_model_liwc.predict(X_test)


#         y_predicted_final = ["xx-24"]*len(y_predicted_rfc)
        
#         # ensemble
#         for i in range(len(y_predicted_final)):
#             if (y_predicted_rfc[i] == y_predicted_rfc_liwc[i]):
#                 y_predicted_final[i] = y_predicted_rfc[i]
#             elif (y_predicted_rfc[i] == y_predicted_svm[i]):
#                 y_predicted_final[i] = y_predicted_rfc[i]
#             elif (y_predicted_rfc_liwc[i] == y_predicted_svm[i]):
#                 y_predicted_final[i] = y_predicted_rfc_liwc[i]

        # adding the gender column predicted by our model to the dataframe
        df_text['age'] = y_predicted_rfc

        # using the ID and age columns in our dataframe to create a dictionary
        results = dict(zip(df_text['userId'], df_text['age']))

        return results
