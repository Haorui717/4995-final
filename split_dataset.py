import os
from shutil import copyfile
from random import shuffle
dataset_dir = '/home/hs3374/celeba/celeba-hq-all'
file_list = os.listdir(dataset_dir)
shuffle(file_list)

for i in range(30000):
    if i < 3000:
        dst_dir = '/home/hs3374/celeba/test'
    else:
        dst_dir = '/home/hs3374/celeba/train'
    copyfile(os.path.join(dataset_dir, file_list[i]),
             os.path.join(dst_dir, file_list[i]))