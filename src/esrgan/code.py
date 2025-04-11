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

# ESRGAN Generator Components
class DenseBlock(nn.Module):
    def __init__(self, num_filters=64, num_layers=5):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(num_filters * i + 64, 64, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
    
    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, dim=1))
            inputs.append(out)
        return torch.cat(inputs, dim=1)

class RRDB(nn.Module):
    def __init__(self, num_filters=64, num_dense_layers=3):
        super(RRDB, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(num_filters) for _ in range(num_dense_layers)])
        self.scale = 0.2
    
    def forward(self, x):
        out = x
        for block in self.dense_blocks:
            residual = block(out)
            out = out + residual * self.scale
        return out * self.scale + x

class ESRGANGenerator(nn.Module):
    def __init__(self, upscale_factor=4, num_filters=64, num_rrdb_blocks=23):
        super(ESRGANGenerator, self).__init__()
        self.initial = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(num_filters) for _ in range(num_rrdb_blocks)])
        self.trunk_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        upsample_layers = []
        for _ in range(int(upscale_factor // 2)):
            upsample_layers += [
                nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(2)
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.final = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        initial = self.initial(x)
        trunk = self.rrdb_blocks(initial)
        trunk = self.trunk_conv(trunk) + initial
        upsampled = self.upsample(trunk)
        return torch.tanh(self.final(upsampled))

# PatchGAN Discriminator (Same as SRGAN)
class PatchGAN(nn.Module):
    def __init__(self):
        super(PatchGAN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)

# Dataset Class (Same as SRGAN)
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

# Vessel Score (Same Dummy Implementation)
def vessel_score(pred, target):
    return torch.abs(pred - target).mean()

# Initialize Models
generator = ESRGANGenerator(upscale_factor=4).to(device)
discriminator = PatchGAN().to(device)

# Loss, Metrics, and Optimizers (Same as SRGAN)
adv_criterion = nn.BCEWithLogitsLoss()
content_criterion = nn.MSELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_loss = lpips.LPIPS(net='alex').to(device)

# Hyperparameters (Same as SRGAN)
lambda_content = 1.0
lambda_adv = 0.001

# Data Preparation (Same as SRGAN)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = FundusDataset(img_dir='../../data/lowres/eyepacs/train', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count())

# Training Loop (Same as SRGAN)
num_epochs = 100
for epoch in range(num_epochs):
    running_d_loss = 0.0
    running_g_total = 0.0
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (lr_imgs, hr_imgs) in loop:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
        
        # Train Discriminator
        optimizer_d.zero_grad()
        fake_imgs = generator(lr_imgs).detach()
        real_preds = discriminator(hr_imgs)
        fake_preds = discriminator(fake_imgs)
        loss_real = adv_criterion(real_preds, torch.ones_like(real_preds))
        loss_fake = adv_criterion(fake_preds, torch.zeros_like(fake_preds))
        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        fake_imgs = generator(lr_imgs)
        fake_preds = discriminator(fake_imgs)
        loss_g_adv = adv_criterion(fake_preds, torch.ones_like(fake_preds))
        loss_content = content_criterion(fake_imgs, hr_imgs)
        total_loss_g = lambda_content * loss_content + lambda_adv * loss_g_adv
        total_loss_g.backward()
        optimizer_g.step()
        
        running_d_loss += loss_d.item()
        running_g_total += total_loss_g.item()
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(D_loss=loss_d.item(), G_total=total_loss_g.item())
    
    avg_d_loss = running_d_loss / len(loader)
    avg_g_total = running_g_total / len(loader)
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {avg_d_loss:.4f}, G Total: {avg_g_total:.4f}")

# Evaluation Loop (Same as SRGAN)
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

# Print final metrics
print("\nESRGAN Model Metrics:")
print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.2f}, LPIPS: {avg_lpips:.2f}, Vessel Score: {avg_vessel:.2f}")