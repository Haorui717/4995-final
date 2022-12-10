import sys
from pathlib import Path
print(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent))  # add project root to path

from models.pSp.psp import pSp
from option.test_psp_option import TestPspOpts


import cv2
import torch
import numpy as np
from PIL import Image
import os

# img = Image.open('img.png')
# cv2.imwrite('test.png', np.array(img))


def tensor2np(tensor):  # rgb -> bgr
    tensor = tensor.squeeze(0) \
        .float().detach().cpu().clamp_(0, 1)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8)
    return img_np


def main(opts):
    psp = pSp(opts)
    latents = []
    for i in range(10):
        noise = torch.randn(1, 512, device=opts.device)
        image_1024, latent = psp.decoder([noise],
                                         input_is_latent=False,
                                         return_latents=True)
        latents.append(latent)
        image_1024 = (image_1024 + 1) / 2
        image_1024_np = tensor2np(image_1024)
        cv2.imwrite(os.path.join(opts.save_dir, f'{i}.png'), image_1024_np)

    for i in range(10):
        image_1024, _ = psp.decoder([latents[i]],
                                         input_is_latent=True,
                                         return_latents=False)
        image_1024 = (image_1024 + 1) / 2
        image_1024_np = tensor2np(image_1024)
        cv2.imwrite(os.path.join('tmp2', f'{i}.png'), image_1024_np)

if __name__ == '__main__':
    opts = TestPspOpts().parse()
    opts.ckptpath = 'checkpoints/psp/psp_ffhq_encode.pt'
    opts.save_dir = 'tmp'
    main(opts)
