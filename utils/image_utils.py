import cv2
import numpy as np
import torch

def read_image(path, grayscale=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not grayscale else img
    return img / 255.0

def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def to_tensor(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return img / 1.0

def denormalize(tensor):
    return tensor.clamp(0, 1).cpu().detach().numpy().transpose(1, 2, 0)
