import os
from re import L
from tqdm import tqdm
import shutil

data_path = '/home/vimlab/Downloads/eth_256_train/eth_256'
train_path = '/home/vimlab/Downloads/eth_256_train/eth_256_train'
val_path = '/home/vimlab/Downloads/eth_256_train/eth_256_val'

files = os.listdir(data_path)

count = 0
for file in tqdm(files):
    # print(os.path.join(data_path, file))
    if count % 10 == 0: 
        shutil.move(os.path.join(data_path, file), os.path.join(val_path, file))
    else:
        shutil.move(os.path.join(data_path, file), os.path.join(train_path, file))
    count += 1