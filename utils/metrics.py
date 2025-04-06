import torch
import torch.nn.functional as F
import lpips
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

lpips_alex = lpips.LPIPS(net='alex').eval()

def psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

def compute_ssim(sr, hr):
    sr_np = sr.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    hr_np = hr.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return ssim(sr_np, hr_np, channel_axis=2, data_range=1.0)

def compute_lpips(sr, hr):
    return lpips_alex(sr, hr).mean().item()
