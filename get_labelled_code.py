import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from models.pSp.psp import pSp
from models.classifier import Classifier
from option.gen_code_option import GenCodeOption

class LatentDataGenerator:
    def __init__(self, opts):
        self.opts = opts
        self.classifier = Classifier()
        self.classifier = self.classifier.to(opts.device)
        self.psp = pSp(opts)
        self.filenames = [file_name for file_name in os.listdir(opts.cls_ckpt_dir)]
        self.filenames.sort()
        os.makedirs(self.opts.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.opts.save_dir, 'label'), exist_ok=True)
        # self.attr_idx_list = [21]
        # self.attr_idx_list = range(40)

    def gen_labelled_latents(self):
        # for filename in os.listdir(self.opts.cls_ckpt_dir):
        for j in range(40):
            print(j)
            filename = self.filenames[j]
            attr_name = os.path.splitext(filename)[0]
            weight_path = os.path.join(self.opts.cls_ckpt_dir, filename)
            self.classifier.load_state_dict(torch.load(weight_path))
            self.classifier.eval()
            self.psp.eval()
            latents = []
            labels = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(self.opts.num_codes // self.opts.batch_size)):
                    noise = torch.randn(self.opts.batch_size, 512, device=self.opts.device)
                    image_1024, latent = self.psp.decoder([noise],
                                                          input_is_latent=False,
                                                          return_latents=True)
                    latents.append(latent.cpu())
                    image_224 = F.interpolate(image_1024, (224, 224), mode='area')
                    label = self.classifier(image_224).squeeze()
                    label[label > 0.5], label[label <= 0.5] = 1, 0
                    labels.append(label.cpu())
            latents_np = torch.cat(latents, dim=0).cpu().numpy()
            labels_np = torch.cat(labels, dim=0).cpu().numpy()
            # latents_np = np.concatenate(latents, axis=0)
            # labels_np = np.concatenate(labels, axis=0)
            np.save(os.path.join(self.opts.save_dir, 'latent', f'{attr_name}'), latents_np)
            np.save(os.path.join(self.opts.save_dir, 'label',  f'{attr_name}'), labels_np)

    def gen_labelled_v2(self):
        latents = []
        labels = [[] for i in range(len(self.filenames))]
        with torch.no_grad():
            for i in tqdm.tqdm(range(self.opts.num_codes // self.opts.batch_size)):
                noise = torch.randn(self.opts.batch_size, 512, device=self.opts.device)
                image_1024, latent = self.psp.decoder([noise],
                                                      input_is_latent=False,
                                                      return_latents=True)
                image_224 = F.interpolate(image_1024, (224, 224), mode='area')
                latents.append(latent.cpu())
                for j in range(len(self.filenames)):
                    filename = self.filenames[j]
                    weight_path = os.path.join(self.opts.cls_ckpt_dir, filename)
                    self.classifier.load_state_dict(torch.load(weight_path))
                    self.classifier.eval()
                    label = self.classifier(image_224).squeeze()
                    label[label > 0.5], label[label <= 0.5] = 1, 0
                    labels[j].append(label.cpu())

        latents_np = torch.cat(latents, dim=0).cpu().numpy()
        labels_np = [torch.cat(label, dim=0).cpu().numpy() for label in labels]


        np.save(os.path.join(self.opts.save_dir, 'latent.npy'), latents_np)
        for i in range(len(labels_np)):
            filename = self.filenames[i]
            attr_name = os.path.splitext(filename)[0]
            np.save(os.path.join(self.opts.save_dir, 'label', f'{attr_name}.npy'), labels_np[j])

if __name__ == "__main__":
    opts = GenCodeOption().parse()
    latentDataGenerator = LatentDataGenerator(opts)
    # latentDataGenerator.gen_labelled_latents()
    latentDataGenerator.gen_labelled_v2()
