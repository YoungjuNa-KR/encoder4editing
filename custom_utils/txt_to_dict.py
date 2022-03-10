import os
import pickle
from tqdm import tqdm

with open("./mpii_labels.txt") as f:
    lines = f.readlines()
    

dataset = 'MPII'

label_dict = {}

if "ETH" in dataset:
    for line in tqdm(lines):
        name = line.split(' ')[0]
        gaze_label = line.split(' ')[2].split(',')
        gaze_label[-1] = gaze_label[-1][:-2]
        for i, l in enumerate(gaze_label):
            gaze_label[i] = float(l)
        label_dict[name] = gaze_label  
elif "MPII" in dataset:
    for line in tqdm(lines):
        name = line.split(' ')[0]
        gaze_label = [0]*2
        gaze_label[0] = float(line.split(' ')[1])
        gaze_label[1] = float(line.split(' ')[2])

        label_dict[name] = gaze_label


file_name = dataset + 'label_dict.pickle'

with open(file_name, "wb") as f:
    pickle.dump(label_dict, f)


with open(file_name, "rb") as f:
    label_dict = pickle.load(f)


print(label_dict['p12_1939'])
print(type(label_dict['p12_1939']))