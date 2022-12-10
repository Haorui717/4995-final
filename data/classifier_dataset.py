import os
import re

import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    assert os.path.isdir(directory), 'dataset does not exist!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])

class ClassifierDataset(Dataset):
    def  __init__(self, data_dir, image_list_path=None, list_attr_celeba_path=None, img_size=224):
        self.img_paths = [path for path in list_pictures(data_dir)]
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if image_list_path is not None and list_attr_celeba_path is not None:
            self.have_label = True
            with open(image_list_path, 'r') as f:
                lines = f.readlines()
                lines = [line.split() for line in lines]

            idx_list = []
            for line in lines[1:]:
                idx_list.append(int(line[1]) + 1)

            with open(list_attr_celeba_path, 'r') as f:
                lines = f.readlines()
                lines = [line.split() for line in lines]

            self.attrs = [attr for attr in lines[1]]
            self.label_list = []
            for idx in idx_list:
                line = lines[idx + 1]
                self.label_list.append(
                    [1 if int(i) == 1 else 0 for i in line[1:]]
                )
            self.label_list = torch.Tensor(self.label_list)

        else:
            self.have_label = False
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        file_idx = int(os.path.splitext(os.path.basename(path))[0])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        img_tensor = self.transform(img)
        if self.have_label:
            return img_tensor, self.label_list[file_idx]
        else:
            return img_tensor
