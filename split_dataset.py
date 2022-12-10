import os
from shutil import copyfile
from random import shuffle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)

def main(opts):
    dataset_dir = opts.dataset_dir
    test_dir, train_dir = opts.test_dir, opts.train_dir
    os.makedirs(test_dir, exist_ok=False)
    os.makedirs(train_dir, exist_ok=False)
    file_list = os.listdir(dataset_dir)
    shuffle(file_list)

    for i in range(30000):
        if i < 3000:
            dst_dir = test_dir
        else:
            dst_dir = train_dir
        copyfile(os.path.join(dataset_dir, file_list[i]),
                 os.path.join(dst_dir, file_list[i]))

if __name__ == '__main__':
    opts = parser.parse_args()
    main(opts)