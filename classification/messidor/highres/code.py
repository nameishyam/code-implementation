import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler  # For mixed precision

# Define image transformations
SIZE, _ = Image.open("./highres/test/3/069f43616fab-600-FA-HFA.jp").size  # Image size taken from a sample image as the folder conatins all of the images with the same size
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

# Set data directory paths
root = './highres/'
train_directory = root + 'train'
val_directory = root + 'val'
test_directory = root + 'test'

# Load data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'val': datasets.ImageFolder(root=val_directory, transform=image_transforms['val']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

# Function to count images per class in a dataset
def count_images_per_class(dataset):
    class_counts = {}
    for _, label in dataset.imgs:
        class_name = dataset.classes[label]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    return class_counts

# Print the number of images in each class for train, val, and test subfolders
for phase in ['train', 'val', 'test']:
    print(f"\n{phase.capitalize()} set class distribution:")
    class_counts = count_images_per_class(data[phase])
    for class_name, count in class_counts.items():
        print(f"Class {class_name}: {count} images")

# Data sizes
train_data_size = len(data['train'])
val_data_size = len(data['val'])
test_data_size = len(data['test'])

# Print total data sizes
print(f"\nTrain data size: {train_data_size}")
print(f"Validation data size: {val_data_size}")
print(f"Test data size: {test_data_size}")

# Batch size (increased for GPU utilization)
batch_size = 32  # Adjust based on GPU memory (e.g., 64 if possible)

# Create data loaders with increased num_workers
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
val_data = DataLoader(data['val'], batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

# Define a simple CNN module
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, device, loss, optimizer, and scheduler
model = SimpleCNN(num_classes=5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Training function with validation, model saving, and graph plotting
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():  # Mixed precision
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_data_size
        epoch_acc = running_corrects.double() / train_data_size
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_data:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with autocast():  # Mixed precision
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / val_data_size
        epoch_val_acc = val_corrects.double() / val_data_size
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc.item())
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

        # Save best model based on validation accuracy
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'best_model_highres.pth')
            print("Saved best model with Val Acc: {:.4f}".format(best_acc))

    # Plot accuracy and loss graphs
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Test function with confusion matrix using Seaborn
def test_model(model):
    model.eval()
    test_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast():  # Mixed precision
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            test_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = test_corrects.double() / test_data_size
    print(f'Test Accuracy: {test_acc:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = data['test'].classes

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Train the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=100)

# Test the model
test_model(model)