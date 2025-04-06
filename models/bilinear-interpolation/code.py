import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import lpips
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
import os
import torch.nn.functional as F

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class FundusDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        hr_image = Image.open(img_path).convert('RGB')
        lr_image = hr_image.resize((128, 128), Image.BICUBIC)  # Downsample to 128x128
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        return lr_image, hr_image

# Vessel Score (Dummy Implementation)
def vessel_score(pred, target):
    return torch.abs(pred - target).mean()

# Metrics
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_loss = lpips.LPIPS(net='alex').to(device)

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = FundusDataset(img_dir='./images', transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())

# Load VGG19 for content loss
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Function to get features from a specific layer
def get_features(x, model, layer_idx):
    for i, layer in enumerate(model):
        x = layer(x)
        if i == layer_idx:
            return x
    return x

# Evaluate Bicubic Upscaling
running_psnr = 0.0
running_ssim = 0.0
running_lpips = 0.0
running_vessel = 0.0
running_content = 0.0
num_batches = 0

# Define normalization parameters outside the loop for efficiency
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)

with torch.no_grad():
    for lr_imgs, hr_imgs in loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        # Upscale LR to HR size using Bicubic interpolation
        bicubic_upscaled = transforms.Resize(
            hr_imgs.shape[-2:], interpolation=transforms.InterpolationMode.BICUBIC
        )(lr_imgs)
        
        # Compute content loss using normalized images (VGG expects normalized inputs)
        pred_features = get_features(bicubic_upscaled, vgg, layer_idx=21)  # Conv4_2 layer
        target_features = get_features(hr_imgs, vgg, layer_idx=21)
        content_loss = F.mse_loss(pred_features, target_features)
        running_content += content_loss.item()
        
        # Denormalize for other metric computations
        bicubic_upscaled = bicubic_upscaled * std + mean
        hr_imgs = hr_imgs * std + mean
        
        # Compute metrics
        running_psnr += psnr_metric(bicubic_upscaled, hr_imgs).item()
        running_ssim += ssim_metric(bicubic_upscaled, hr_imgs).item()
        running_lpips += lpips_loss(bicubic_upscaled, hr_imgs).mean().item()
        running_vessel += vessel_score(bicubic_upscaled, hr_imgs).item()
        num_batches += 1

# Average metrics
avg_psnr = running_psnr / num_batches
avg_ssim = running_ssim / num_batches
avg_lpips = running_lpips / num_batches
avg_vessel = running_vessel / num_batches
avg_content = running_content / num_batches

# Print final metrics
print("\nBicubic Model Metrics:")
print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.2f}, LPIPS: {avg_lpips:.2f}, Vessel Score: {avg_vessel:.2f}, Content Loss: {avg_content:.4f}")