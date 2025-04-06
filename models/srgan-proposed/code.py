import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from pytorch_msssim import SSIM
import lpips
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image
import os
import itertools

# Placeholder for DRNet (assumed to be imported from an external file)
from drnet import DRNet

# Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

# Generator (SRGAN)
class SRGenerator(nn.Module):
    def __init__(self, drnet_params: dict, upscale_factor=4, additional_blocks=2):
        super(SRGenerator, self).__init__()
        self.feature_extractor = DRNet(**drnet_params).encoder
        base_channels = drnet_params.get('base_channels', 64)
        
        self.additional_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(additional_blocks)]
        )
        
        upsample_layers = []
        current_channels = base_channels
        while upscale_factor > 1:
            upsample_layers += [
                nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]
            current_channels = current_channels // 2
            upscale_factor = upscale_factor // 2
        upsample_layers += [nn.Conv2d(current_channels, 3, kernel_size=3, padding=1)]
        self.upsample = nn.Sequential(*upsample_layers)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.additional_blocks(features)
        return self.upsample(features)

# Discriminator (PatchGAN)
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

# Vessel Loss (Placeholder)
def vessel_loss(sr, hr):
    return torch.abs(sr - hr).mean()

# Training Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DRNet parameters configuration
drnet_config = {
    'in_channels': 3,
    'base_channels': 64,
    'encoder_blocks': 4
}

# Hyperparameter Search Space for Tuning
param_grid = {
    'lambda_content': [0.5, 1.0, 1.5],
    'lambda_adv': [0.0005, 0.001, 0.005],
    'lambda_ssim': [0.05, 0.1, 0.2],
    'lambda_lpips': [0.05, 0.1, 0.2],
    'lambda_vessel': [0.3, 0.5, 0.7]
}

# Training and Evaluation Function
def train_and_evaluate(lambda_content, lambda_adv, lambda_ssim, lambda_lpips, lambda_vessel):
    # Model Initialization
    generator = SRGenerator(drnet_params=drnet_config, upscale_factor=4, additional_blocks=2).to(device)
    discriminator = PatchGAN().to(device)
    
    # Load pre-trained DRNet weights if available
    try:
        generator.feature_extractor.load_state_dict(torch.load('DRNet.pth'), strict=False)
        print("Loaded DRNet weights successfully")
    except Exception as e:
        print(f"Failed to load weights: {e}")
    
    # Freeze DRNet parameters
    for param in generator.feature_extractor.parameters():
        param.requires_grad = False
    
    # Loss, Metrics, and Optimizers
    adv_criterion = nn.BCEWithLogitsLoss()
    content_criterion = nn.MSELoss()
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FundusDataset(img_dir='./images', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count())
    
    # Training Loop
    num_epochs = 10  # Reduced for tuning; increase for final training
    for epoch in range(num_epochs):
        running_d_loss = 0.0
        running_g_total = 0.0
        running_content = 0.0
        running_adv = 0.0
        running_ssim = 0.0
        running_lpips = 0.0
        running_vessel = 0.0
        
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
            loss_ssim = 1 - ssim_module(fake_imgs, hr_imgs)
            loss_lpips_val = lpips_loss(fake_imgs, hr_imgs).mean()
            loss_vessel_val = vessel_loss(fake_imgs, hr_imgs)
            total_loss_g = (
                lambda_content * loss_content +
                lambda_adv * loss_g_adv +
                lambda_ssim * loss_ssim +
                lambda_lpips * loss_lpips_val +
                lambda_vessel * loss_vessel_val
            )
            total_loss_g.backward()
            optimizer_g.step()
            
            # Update running losses
            running_d_loss += loss_d.item()
            running_g_total += total_loss_g.item()
            running_content += loss_content.item()
            running_adv += loss_g_adv.item()
            running_ssim += loss_ssim.item()
            running_lpips += loss_lpips_val.item()
            running_vessel += loss_vessel_val.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(D_loss=loss_d.item(), G_total=total_loss_g.item(), Content=loss_content.item())
        
        # Calculate average losses for the epoch
        avg_d_loss = running_d_loss / len(loader)
        avg_g_total = running_g_total / len(loader)
        avg_content = running_content / len(loader)
        avg_adv = running_adv / len(loader)
        avg_ssim = running_ssim / len(loader)
        avg_lpips = running_lpips / len(loader)
        avg_vessel = running_vessel / len(loader)

        # if (epoch == num_epochs - 1):
        #     # Print epoch-wise losses
        #     print(f"\nEpoch {epoch+1}/{num_epochs}")
        #     print(f"D Loss: {avg_d_loss:.4f} | G Total: {avg_g_total:.4f}")
        #     print(f"  Content: {avg_content:.4f}")
        #     print(f"  Adv: {avg_adv:.4f}")
        #     print(f"  SSIM: {avg_ssim:.4f}")
        #     print(f"  LPIPS: {avg_lpips:.4f}")
        #     print(f"  Vessel: {avg_vessel:.4f}")
    
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
            running_vessel += vessel_loss(fake_imgs, hr_imgs).item()
            num_batches += 1
    
    # Average metrics
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches
    avg_lpips = running_lpips / num_batches
    avg_vessel = running_vessel / num_batches
    
    print(f"\nEvaluation for lambda_content={lambda_content}, lambda_adv={lambda_adv}, lambda_ssim={lambda_ssim}, lambda_lpips={lambda_lpips}, lambda_vessel={lambda_vessel}")
    print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.2f}, LPIPS: {avg_lpips:.2f}, Vessel Score: {avg_vessel:.2f}")
    
    return avg_psnr, avg_ssim, avg_lpips, avg_vessel, generator, discriminator

# Grid Search for Hyperparameter Tuning
def grid_search():
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    best_psnr = 0.0
    best_params = None
    best_generator = None
    best_discriminator = None
    best_metrics = None
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        lambda_content = params['lambda_content']
        lambda_adv = params['lambda_adv']
        lambda_ssim = params['lambda_ssim']
        lambda_lpips = params['lambda_lpips']
        lambda_vessel = params['lambda_vessel']
        
        print(f"\nTuning with: {params}")
        avg_psnr, avg_ssim, avg_lpips, avg_vessel, generator, discriminator = train_and_evaluate(
            lambda_content, lambda_adv, lambda_ssim, lambda_lpips, lambda_vessel
        )
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_params = params
            best_metrics = (avg_psnr, avg_ssim, avg_lpips, avg_vessel)
            best_generator = generator
            best_discriminator = discriminator
    
    # Print final metrics for the best model
    print("\nProposed Model Metrics (Best Configuration):")
    print(f"PSNR: {best_metrics[0]:.2f}, SSIM: {best_metrics[1]:.2f}, LPIPS: {best_metrics[2]:.2f}, Vessel Score: {best_metrics[3]:.2f}")
    print(f"Best Hyperparameters: {best_params}")
    
    # Save the best models
    torch.save(best_generator.state_dict(), 'SRGAN_Generator_Best.pth')
    torch.save(best_discriminator.state_dict(), 'SRGAN_Discriminator_Best.pth')

# Main Execution
if __name__ == "__main__":
    grid_search()