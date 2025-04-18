# Import the required libraries:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Unzip the dataset:
!mkdir "custom_stl10_subset"

!unzip -q "/content/custom_stl10_subset.zip" -d "/content/custom_stl10_subset"

# Transform (convert images to tensors)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root='/content/custom_stl10_subset', transform=transform)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Simple Autoencoder Definition
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Model, Loss, Optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot Training Loss
plt.figure(figsize=(6,4))
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()

# Visualize Original and Reconstructed Images
def show_images(original, reconstructed, n=6):
    original = original[:n].cpu()
    reconstructed = reconstructed[:n].cpu()

    fig, axes = plt.subplots(2, n, figsize=(15,4))
    for i in range(n):
        axes[0, i].imshow(np.transpose(original[i], (1,2,0)))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(np.transpose(reconstructed[i].detach(), (1,2,0)))
        axes[1, i].set_title('Compressed')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# Evaluate on a sample batch
sample_imgs, _ = next(iter(loader))
sample_imgs = sample_imgs.to(device)
with torch.no_grad():
    reconstructed_imgs = model(sample_imgs)

show_images(sample_imgs, reconstructed_imgs)

# Compute overall MSE on the entire dataset
model.eval()
total_mse = 0.0
with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        total_mse += criterion(outputs, imgs).item()

average_mse = total_mse / len(loader)
print(f"Average MSE on dataset: {average_mse:.4f}")