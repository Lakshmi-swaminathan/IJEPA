# -*- coding: utf-8 -*-
"""IJEPA_Alexnet_Training_v1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lO31J9sHaMNMgEz0kE_2xvZOBerDg64J

## Training Alexnet Model with IJEPA

### Importing Libraries
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import alexnet

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""### Importing the CIFAR Dataset"""

# Load CIFAR dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224 for AlexNet compatibility
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

"""### Defining the Major Modules in IJEPA Architecture
- Encoder
- Predictor
- TargetEncoder
"""

# Encoder class using AlexNet
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = alexnet(pretrained=False)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])  # Remove last layer

    def forward(self, x):
        return self.model(x)

# Target Encoder class with output dimension of 1024
class TargetEncoder(nn.Module):
    def __init__(self):
        super(TargetEncoder, self).__init__()
        self.model = alexnet(pretrained=True)
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1],  # Remove last layer
            nn.Linear(4096, 1024)  # Change to output 1024 dimensions
        )

    def forward(self, x):
        with torch.no_grad():  # Ensure target encoder doesn’t get updated
            return self.model(x)

# Predictor class for transforming context embeddings to predicted target embeddings
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 1024),  # Match context output size
            nn.ReLU(),
            nn.Linear(1024, 1024)   # Output size same as target encoder
        )

    def forward(self, x):
        return self.layers(x)

"""### Defining the Masking Strategy
- Random Masking
"""

import random
import torch

# Masking Strategy class for masking random regions in the input images
class MaskingStrategy:
    def __init__(self):
        pass  # No fixed mask size; we'll randomize it dynamically

    def apply_mask(self, images):
        # Clone images to avoid modifying the original input
        masked_images = images.clone()

        # Get image dimensions
        _, _, h, w = images.shape

        # Randomly determine the mask size in the range [10, 30]
        mask_h = random.randint(30, 80)
        mask_w = random.randint(30, 80)

        # Ensure the random mask position fits within the image boundaries
        x = random.randint(0, h - mask_h)
        y = random.randint(0, w - mask_w)

        # Apply mask by setting a random region to zero
        masked_images[:, :, x:x + mask_h, y:y + mask_w] = 0
        return masked_images

# Take a sample image and apply the masking strategy
sample_image = next(iter(train_loader))[0]
masking_strategy = MaskingStrategy()
masked_image = masking_strategy.apply_mask(sample_image)

# use matplotlib
import matplotlib.pyplot as plt

# create a 2:1 plot
fig, axs = plt.subplots(1, 2, figsize=(6,3))

# Add sample image
axs[0].imshow(sample_image[0].permute(1, 2, 0))
axs[0].set_title('Original Image')

# Add masked image
axs[1].imshow(masked_image[0].permute(1, 2, 0))



"""### IJEPA Model"""

# IJEPA Model that combines Encoder, TargetEncoder, Predictor, and MaskingStrategy
class IJEPA_Model(nn.Module):
    def __init__(self):
        super(IJEPA_Model, self).__init__()
        self.encoder = Encoder()
        self.target_encoder = TargetEncoder()
        self.predictor = Predictor()
        self.masking_strategy = MaskingStrategy()

    def forward(self, images):
        # Context embedding from masked images
        context_embedding = self.encoder(images)

        # Target embedding from original images
        target_embedding = self.target_encoder(images)

        # Predict masked embeddings from context embedding
        predicted_embedding = self.predictor(context_embedding)

        return predicted_embedding, target_embedding

"""### Training Loop"""

import os

# Training loop with multiple random masks per image
def train(model, dataloader, epochs=10, lr=1e-2, num_masks=4, save_interval=2, checkpoint_dir='checkpoints'):
    model.train()
    model.to(device)  # Move model to GPU
    criterion = nn.MSELoss()  # Cosine similarity for embedding loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # List to store loss values for plotting
    epoch_losses = []

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Training Started...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, _ in dataloader:
            images = images.to(device)  # Move data to GPU

            # Accumulate loss for multiple masked versions
            batch_loss = 0.0

            for _ in range(num_masks):
                # Apply random masking to the images
                masked_images = model.masking_strategy.apply_mask(images)

                # Forward pass
                predicted_embedding, target_embedding = model(masked_images)

                # Cosine similarity loss (target label for similarity is 1)
                target = torch.ones(predicted_embedding.size(0)).to(device)
                loss = criterion(predicted_embedding, target_embedding)

                # Accumulate the loss
                batch_loss += loss

            # Average the loss over multiple masks
            batch_loss /= num_masks

            # Backward pass and optimization
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

        # Diving by batch size
        epoch_loss /= len(dataloader)

        # Append average epoch loss
        epoch_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Save checkpoint every few epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    return epoch_losses  # Return the list of epoch losses

"""### Start Training"""

# Initialize model and start training
model = IJEPA_Model()
losses = train(model, train_loader, epochs=30)

"""### Plotting Loss over Epochs"""

import matplotlib.pyplot as plt

# Plotting the loss graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, len(losses) + 1))  # Set x-ticks to be epoch numbers
plt.grid()
plt.show()

import os
from google.colab import files

# Specify your checkpoint directory
checkpoint_dir = '/content/checkpoints'

# List all checkpoint files
all_checkpoints = sorted(os.listdir(checkpoint_dir))

# Get the latest checkpoint file
latest_checkpoint = all_checkpoints[-1] if all_checkpoints else None

if latest_checkpoint:
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Downloading {latest_checkpoint}...")
    files.download(checkpoint_path)  # Download the file
else:
    print("No checkpoint files found.")