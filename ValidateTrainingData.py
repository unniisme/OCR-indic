import pandas as pd
import sys

fileName = sys.argv[1]

data = pd.read_csv(fileName)

print("Average Time :", data["time"].mean(), "sec")

data = data[data["label"].str.isupper()]
data_correct = data[data["label"] == data["predicted_label"]]

print("accuracy :", 100*round(len(data_correct)/len(data),3), "%")
