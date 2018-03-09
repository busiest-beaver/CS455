import matplotlib.pyplot as plt
import pandas as pd

def plotPersonality():
    traits = ["ope", "neu", "ext", "agr", "con"]
    traitNames = ["OPENNESS", "NEUROTIC", "EXTROVERT", "AGREEABLE", "CONSCIENTIOUS"]
    for i in range(len(traits)):
        df = pd.read_csv("scores"+traits[i]+".csv")
        df.boxplot(column='scores', by='algorithms')
        plt.title(traitNames[i])
        plt.ylabel('RMSE (lower is better)')
        plt.xlabel("ALGORITHMS")
        plt.xticks(rotation=65)    
        plt.show()

def plotGender():
    df = pd.read_csv("scoresGender.csv")
    df.boxplot(column="scores", by="algorithms")
    plt.title("Gender")
    plt.ylabel('ACCURACY (higher is better)')
    plt.xlabel("ALGORITHMS")
    plt.xticks(rotation=65)
    plt.show()

plotPersonality()
# plotGender()