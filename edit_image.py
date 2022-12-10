import os

import numpy as np
import torch
import tqdm
from torch.utils.data import dataloader
from models.pSp.psp import pSp
from data.classifier_dataset import ClassifierDataset
from util import tensor2np, save_img
from option.edit_image_option import EditImageOption

class ImageEditor:
    def __init__(self, opts):
        self.opts = opts
        self.psp = pSp(opts)
        if opts.real:
            self.dataset = ClassifierDataset(opts.image_path,
                                             img_size=256)
            self.dataloader = dataloader.DataLoader(self.dataset,
                                                    shuffle=True,
                                                    batch_size=1)
        self.attr_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in os.listdir(opts.direction_dir)]
        self.attr_names.sort()
        direction_paths = [os.path.join(opts.direction_dir, path) for path in os.listdir(opts.direction_dir)]
        direction_paths.sort()
        self.directions = [np.load(path) for path in direction_paths]
        self.directions = [direction/np.sqrt((direction*direction).sum()) for direction in self.directions]
        self.directions = [torch.from_numpy(direction).float().to(opts.device) for direction in self.directions]
        self.directions = torch.cat([direction.unsqueeze(0) for direction in self.directions], dim=0)


    def edit_real(self):
        print('real')
        direction = self.directions[self.opts.attr_idx]
        attr_name = self.attr_names[self.opts.attr_idx]
        for batch, origin in tqdm.tqdm(enumerate(self.dataloader)):
            if batch > self.opts.num_edit:
                break
            origin = origin.to(self.opts.device)
            origin_np = tensor2np((origin + 1) / 2)
            save_list = [origin_np]
            _, latent = self.psp(origin, return_latents=True)
            for i in np.linspace(-30, 30, 7):
                edit, _ = self.psp.decoder([latent + i*direction],
                                           input_is_latent=True,
                                           return_latents=False)
                edit = self.psp.face_pool(edit)
                edit_np = tensor2np((edit + 1) / 2)
                save_list.append(edit_np)
            img_save = np.concatenate(save_list, axis=1)
            save_img(os.path.join(self.opts.save_dir, 'real', attr_name), f'{attr_name}_{batch}.png', img_save)

    def edit_fake(self):
        print('fake')
        direction = self.directions[self.opts.attr_idx]
        attr_name = self.attr_names[self.opts.attr_idx]
        for i in tqdm.tqdm(range(self.opts.num_edit)):
            noise = torch.randn(1, 512, device=self.opts.device)
            _, latent = self.psp.decoder([noise],
                                    input_is_latent=False,
                                    return_latents=True)
            save_list = []
            for j in np.linspace(-30, 30, 7):
                edit, _ = self.psp.decoder([latent + j*direction],
                                           input_is_latent=True,
                                           return_latents=False)
                edit = self.psp.face_pool(edit)
                edit_np = tensor2np((edit + 1) / 2)
                save_list.append(edit_np)
            img_save = np.concatenate(save_list, axis=1)
            save_img(os.path.join(self.opts.save_dir, 'fake', attr_name), f'{attr_name}_{i}.png', img_save)

if __name__ == '__main__':
    opts = EditImageOption().parse()
    imageEditor = ImageEditor(opts)
    if opts.real:
        imageEditor.edit_real()
    else:
        imageEditor.edit_fake()