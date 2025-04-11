import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# ====================== Supporting Modules (Unchanged) ====================== #
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

# ====================== Modified ResidualBlock with Dilated Convolutions ====================== #
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=2):
        super().__init__()
        # Adjust padding to account for dilation
        padding = dilation * ((kernel_size - 1) // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation),
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

# ====================== Modified DRNet for Feature Extraction ====================== #
class DRNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, encoder_blocks=4):
        super().__init__()
        # Encoder with ResidualBlocks incorporating dilated convolutions
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            *[ResidualBlock(base_channels, dilation=2) for _ in range(encoder_blocks)],
            MultiScaleProcessor(base_channels)
        )

    def forward(self, x):
        features = self.encoder(x)
        return features

# ====================== Dataset (Adjusted) ====================== #
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
        
        return scaled_images  # Only return scaled images, no target needed

# ====================== Training/Inference Example ====================== #
def extract_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = MultiScaleFundusDataset(img_dir='../../data/aptos')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=os.cpu_count())
    
    # Initialize model
    model = DRNet().to(device)
    # Optionally load pre-trained weights if available
    # model.load_state_dict(torch.load('best_drnet.pth'))
    model.eval()
    
    # Feature extraction
    with torch.no_grad():
        for scales in data_loader:
            scales = [s.to(device) for s in scales]
            for scale in scales:
                features = model(scale)
                print(f"Features shape for this scale: {features.shape}")
                # Use features for downstream tasks (e.g., save, classify, etc.)
                # Example: features is [batch_size, base_channels, H, W]

if __name__ == "__main__":
    extract_features()
