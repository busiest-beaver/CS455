import os
import pandas as pd

ages = ["xx-24", "25-34", "35-49", "50-xx"]

df = pd.read_csv("process.csv")

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

df.to_csv('processedAge.csv')