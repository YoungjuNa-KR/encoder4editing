import os
import shutil
from tqdm import tqdm

file_path = '/home/vimlab/Downloads/eth_256_edit/train'

move_path = '/home/vimlab/Downloads/eth_256_from_100'


files = os.listdir(file_path)

count = 0
for file in tqdm(files):
    if int(file.split('_')[0]) >= 100:
        count += 1
        # print(file)
        src = os.path.join(file_path, file)
        dst = os.path.join(move_path, file)
        shutil.copy(src, dst)

print(count)