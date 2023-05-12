from ImageTemplate import Template
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def strList_to_list(s):
    s = s.strip('[]')  # remove square brackets
    floats = s.split()  # split string on comma
    return [float(x) for x in floats]

def group_by_value(sums):
    # Truncate values to 2 digits
    sums = [(round(value, 2), name) for value, name in sums]
    # Group tuples by value
    groups = {}
    for value, name in sums:
        if value in groups:
            groups[value].append(name)
        else:
            groups[value] = [name]

    # Get one name from each value bucket
    result = []
    for value, names in groups.items():
        result.append((value, names[0], len(names)))

    return result


np.set_printoptions(precision=3)

templates = {}


_test_path = "./Letters/archive/"

sums = []
data = pd.read_csv(_test_path + "S.csv")
out_data = pd.DataFrame(columns=['image', 'label'])

for index, row in data.iterrows():
    encoding = strList_to_list(row['distanceEncoding'])
    
    # print(index, end = "\r")
    # plt.plot(list(range(len(encoding))), encoding, label=row['image'])

    sums.append((sum(encoding[:125]), row['image']))

sums = sorted(sums)
print("\n".join([str(x) for x in sums]))

groups = group_by_value(sums)
print(groups)
out_data['image'] = [x[1] for x in groups]
out_data['label'] = ['A'] * len(groups)
out_data.to_csv(_test_path + "S_var.csv", index=False)

X = [x[1] for x in sums]
Y = [x[0] for x in sums]
plt.plot(X, Y)


plt.legend()
plt.show()