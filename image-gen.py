import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

# Define your generator architecture (same as used during training)
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Example architecture — replace with your actual one
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, 2, 1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 3, 4, 2, 1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Load the generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load('./model/SRGAN_Generator_Best.pth', map_location=device))
generator.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Change based on model requirements
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Paths
input_folder = './data/lowres/eyepacs'
output_folder = './data/highres/eyepacs'
os.makedirs(output_folder, exist_ok=True)

# Process each image
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, image_name)
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = generator(input_tensor)

        # De-normalize and save
        output_tensor = (output_tensor + 1) / 2  # Convert back to [0,1]
        save_image(output_tensor, os.path.join(output_folder, image_name))

print("Image generation complete.")
