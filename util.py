import os

import numpy as np
import cv2

def tensor2np(tensor):  # rgb -> bgr
    tensor = tensor.squeeze(0) \
        .float().detach().cpu().clamp_(0, 1)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8)
    return img_np

def save_img(dir, filename, img):
    os.makedirs(dir, exist_ok=True)
    cv2.imwrite(os.path.join(dir, filename), img)