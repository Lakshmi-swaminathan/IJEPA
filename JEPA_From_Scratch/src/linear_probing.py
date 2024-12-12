# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import yaml
from src.helper import load_checkpoint_for_linear_probe, init_model
from src.datasets.imagenet1k import make_imagenet1k
from torchvision import transforms

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Device configuration - Use CPU if GPU memory is insufficient
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load configuration from YAML file
config_path = 'configs/in1k_vith14_ep300.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize the model
patch_size = config['mask']['patch_size']
model_name = config['meta']['model_name']
crop_size = config['data']['crop_size']
pred_depth = config['meta']['pred_depth']
pred_emb_dim = config['meta']['pred_emb_dim']

encoder, _ = init_model(
    device=device,
    patch_size=patch_size,
    model_name=model_name,
    crop_size=crop_size,
    pred_depth=pred_depth,
    pred_emb_dim=pred_emb_dim
)

# Load pretrained weights
checkpoint_path = config['meta']['read_checkpoint']
load_checkpoint_for_linear_probe(device, checkpoint_path, encoder)

# Freeze all layers of the encoder
for param in encoder.parameters():
    param.requires_grad = False

# Add a linear layer on top for classification (adjust output features for your target classes)
# Define the number of classes for downstream task
num_classes = 100  # You may need to adjust this based on your downstream dataset
linear_layer = nn.Linear(encoder.embed_dim, num_classes)

# Create a model that combines encoder and linear layer
class LinearProbeModel(nn.Module):
    def __init__(self, encoder, linear_layer):
        super(LinearProbeModel, self).__init__()
        self.encoder = encoder
        self.linear_layer = linear_layer

    def forward(self, x):
        x = self.encoder(x)
        # Average pooling to match dimensions before the linear layer
        x = x.mean(dim=1)  # Assume output shape is (batch_size, num_patches, embed_dim)
        x = self.linear_layer(x)
        return x

# Instantiate the linear probe model
model = LinearProbeModel(encoder, linear_layer).to(device)

# Set the model to training mode
model.train()

# Define the optimizer (only for the linear layer)
optimizer = optim.Adam(model.linear_layer.parameters(), lr=config['optimization']['lr'])

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Load the dataset using configuration from the YAML file
root_path = config['data']['root_path']
image_folder = config['data']['image_folder']
batch_size = config['data']['batch_size']
num_workers = config['data']['num_workers']

# Define the appropriate transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((crop_size, crop_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_, train_loader, _ = make_imagenet1k(
    transform=transform,
    batch_size=batch_size,
    pin_mem=config['data']['pin_mem'],
    training=True,
    num_workers=num_workers,
    root_path=root_path,
    image_folder=image_folder,
    drop_last=True
)

# Train the linear layer (example training loop)
for epoch in range(config['optimization']['epochs']):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    logger.info(f'Epoch [{epoch+1}] Loss: {running_loss/len(train_loader)}')

logger.info('Finished Linear Probing')
