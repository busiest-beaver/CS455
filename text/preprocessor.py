import os
import pandas as pd 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#process and clean text files
def process(content):
    content = content.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(content)
    #filtering stop-words and words with length less than 3
    filteredContent = [w for w in tokens if (not w in stopwords.words('english') and len(w) > 2)]
    return " ".join(filteredContent)

#path to folder containing status txt files
pathToStatus = 'C:/Users/Abdullah/Desktop/455HW/training/text/'
pathToProfile = 'C:/Users/Abdullah/Desktop/455HW/training/profile/profile.csv'
pathToLIWC = 'C:/Users/Abdullah/Desktop/455HW/training/LIWC/LIWC.csv'

df = pd.read_csv(pathToProfile)
df = df.drop('Unnamed: 0', 1)

liwc = pd.read_csv(pathToLIWC)
liwcColumns = liwc.columns

#inserting data from LIWC into the processed file
for i in range(0, len(liwc)):
    id = liwc.iloc[i]['userId']
    for entry in liwcColumns:
        df.loc[df.userid.isin([id]), str(entry)] = liwc.iloc[i][entry]

#adding column for text
df['text'] = ""
#reading in status txt files, and adding them to 'text' column
for fileName in os.listdir(pathToStatus):
    fileContent = open(pathToStatus+fileName).read()
    words = process(fileContent)
    id = os.path.splitext(fileName)[0]
    df.loc[df.userid.isin([id]), 'text'] = words

#adding age labels. this way we can do prediction on continuous and discrete data
ages = ["xx-24", "25-34", "35-49", "50-xx"]
ageLabels = []
for entry in df["age"]:
    if (entry < 25):
        ageLabels.append(ages[0])
    elif (entry < 35):
        ageLabels.append(ages[1])
    elif (entry < 50):
        ageLabels.append(ages[2])
    else:
        ageLabels.append(ages[3])
df["ageLabel"] = ageLabels

df = df.drop('userId', 1)
df.to_csv('processed.csv')
exit()
