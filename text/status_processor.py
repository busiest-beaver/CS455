import os
import pandas as pd #data processing
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import timeit

start = timeit.default_timer()

def process(content):
    content = content.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(content)
    #filtering stop-words and words with length less than 3
    filteredContent = [w for w in tokens if (not w in stopwords.words('english') and len(w) > 2)]
    return " ".join(filteredContent)

#path to folder containing status txt files
pathToStatus = 'training/text/'
pathToProfile = 'training/profile/profile.csv'
pathToLIWC = 'training/LIWC/LIWC.csv'

df = pd.read_csv(pathToProfile)
df = df.drop('Unnamed: 0', 1) #dropping unnamed, number column
#adding column for text
df['text'] = ""

liwc = pd.read_csv(pathToLIWC)

liwcColumns = liwc.columns

#inserting data from LIWC into the processing file
for i in range(0, len(liwc)):
    id = liwc.iloc[i]['userId']
    for entry in liwcColumns:
        df.loc[df.userid.isin([id]), str(entry)] = liwc.iloc[i][entry]

#reading in status txt file
for fileName in os.listdir(pathToStatus):
    fileContent = open(pathToStatus+fileName).read()
    words = process(fileContent)
    id = os.path.splitext(fileName)[0]
    df.loc[df.userid.isin([id]), 'text'] = words

df = df.drop('Unnamed: 0', 1) #dropping unnamed, number column
df = df.drop('userId', 1)
df.to_csv('process.csv')
stop = timeit.default_timer()

print("Start:", start)
print("Stop:", stop)

exit()
