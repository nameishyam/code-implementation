import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lpips
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SRCNN Model with Upsampling
class SRCNN(nn.Module):
    def __init__(self, upscale_factor=4):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3 * upscale_factor * upscale_factor, kernel_size=5, padding=2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.final = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        out = self.pixel_shuffle(out)
        out = self.final(out)
        return self.sigmoid(out)

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
        lr_image = hr_image.resize((128, 128), Image.BICUBIC)
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        return lr_image, hr_image

# Vessel Score (Dummy Implementation)
def vessel_score(pred, target):
    return torch.abs(pred - target).mean()

# Initialize Model
generator = SRCNN(upscale_factor=4).to(device)

# Loss, Metrics, and Optimizer
content_criterion = nn.MSELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_loss = lpips.LPIPS(net='alex').to(device)

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = FundusDataset(img_dir='../../data/lowres/eyepacs', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count())

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (lr_imgs, hr_imgs) in loop:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        optimizer_g.zero_grad()
        fake_imgs = generator(lr_imgs)
        loss_content = content_criterion(fake_imgs, hr_imgs)
        loss_content.backward()
        optimizer_g.step()
        
        running_loss += loss_content.item()
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(Content=loss_content.item())
    
    avg_loss = running_loss / len(loader)
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Content Loss: {avg_loss:.4f}")

# Evaluation Loop
generator.eval()
running_psnr = 0.0
running_ssim = 0.0
running_lpips = 0.0
running_vessel = 0.0
num_batches = 0

with torch.no_grad():
    for lr_imgs, hr_imgs in loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        fake_imgs = generator(lr_imgs)
        
        # Denormalize for metric computation
        fake_imgs = fake_imgs * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        hr_imgs = hr_imgs * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        
        # Compute metrics
        running_psnr += psnr_metric(fake_imgs, hr_imgs).item()
        running_ssim += ssim_metric(fake_imgs, hr_imgs).item()
        running_lpips += lpips_loss(fake_imgs, hr_imgs).mean().item()
        running_vessel += vessel_score(fake_imgs, hr_imgs).item()
        num_batches += 1

# Average metrics
avg_psnr = running_psnr / num_batches
avg_ssim = running_ssim / num_batches
avg_lpips = running_lpips / num_batches
avg_vessel = running_vessel / num_batches

# Print final metrics (simulating table values)
print("\nSRCNN Model Metrics:")
print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.2f}, LPIPS: {avg_lpips:.2f}, Vessel Score: {avg_vessel:.2f}")