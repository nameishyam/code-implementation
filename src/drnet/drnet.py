import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        max = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg + max)
        return x * attention.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg, max], dim=1))
        return x * self.sigmoid(attention)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels)
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.ca(x)
        x = self.sa(x)
        return torch.relu(residual + x)

class MultiScaleProcessor(nn.Module):
    def __init__(self, channels, scales=[0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.downsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ) for _ in scales
        ])
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ) for _ in scales
        ])

    def forward(self, x):
        features = [x]
        for downsample in self.downsample:
            features.append(downsample(features[-1]))
        for i in range(len(self.scales)-1, -1, -1):
            features[i] = features[i] + self.upsample[i](features[i+1])
        return features[0]

class DRNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, encoder_blocks=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            *[ResidualBlock(base_channels) for _ in range(encoder_blocks)],
            MultiScaleProcessor(base_channels)
        )

    def forward(self, x):
        return self.encoder(x)
class MultiScaleFundusDataset(Dataset):
    def __init__(self, img_dir, base_size=128, scales=[1.0, 0.5, 0.25]):
        self.img_dir = img_dir
        self.base_size = base_size
        self.scales = scales
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        scaled_images = []
        for scale in self.scales:
            size = int(self.base_size * scale)
            scaled_img = transforms.Resize((size, size))(image)
            scaled_img = self.transform(scaled_img)
            scaled_images.append(scaled_img)
        return scaled_images

def visualize_image_with_features(original, features, num_channels=5):
    original = original.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original = np.clip(std * original + mean, 0, 1)

    features = features.detach().cpu()
    features_min, features_max = features.min(), features.max()
    features = (features - features_min) / (features_max - features_min)

    fig, axes = plt.subplots(1, num_channels + 1, figsize=(18, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(num_channels):
        axes[i + 1].imshow(features[i].numpy(), cmap='viridis')
        axes[i + 1].set_title(f"Channel {i+1}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

def extract_and_visualize_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = MultiScaleFundusDataset(img_dir='../../data/aptos')
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=os.cpu_count())
    
    model = DRNet().to(device)
    model.eval()
    
    with torch.no_grad():
        for i, scales in enumerate(data_loader):
            if i >= 1:
                break

            largest_scale = scales[0].to(device)
            features = model(largest_scale)

            print("Original Images and Corresponding Feature Maps:")
            for j in range(min(5, largest_scale.size(0))):
                visualize_image_with_features(scales[0][j], features[j])

if __name__ == "__main__":
    extract_and_visualize_features()
