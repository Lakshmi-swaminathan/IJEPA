import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the JEPA checkpoint
checkpoint_path = '/scratch/gilbreth/abelde/DL_Research/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar'
output_dir = '/scratch/gilbreth/abelde/DL_Research/results/celeb'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

try:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(f"Checkpoint loaded successfully: {list(checkpoint.keys())}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit()

# Extract encoder weights
encoder_weights = checkpoint.get("encoder", None)
if encoder_weights is None:
    print("No encoder weights found in the checkpoint.")
    exit()

# Define the encoder
class EncoderModel(nn.Module):
    def __init__(self, pretrained_weights):
        super().__init__()
        self.encoder = models.vit_h_14(weights=None)  # Vision Transformer without pretrained weights
        self.encoder.load_state_dict(pretrained_weights, strict=False)
        self.encoder.heads.head = nn.Identity()  # Remove the classification head

    def forward(self, x):
        return self.encoder(x)

# Define the decoder
class DecoderModel(nn.Module):
    def __init__(self, latent_dim=1280):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 512),  # Expand embedding to 8x8 feature map
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 8, 8)),  # Reshape to (512, 8, 8)

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample to (16x16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample to (32x32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Upsample to (64x64)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),    # Upsample to (128x128)
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, x):
        return self.decoder(x)

# Prepare CelebA dataset
def get_celeba_dataloader(batch_size=32, train=True):
    split = 'train' if train else 'valid'
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # Center crop to square
        transforms.Resize((224, 224)),  # Resize to 224x224 for the encoder
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CelebA(root='./data', split=split, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)

# Train the autoencoder
def train_autoencoder(encoder, decoder, dataloader, optimizer, criterion, num_epochs, device):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()  # Freeze the encoder
    decoder.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Wrap the dataloader with tqdm for progress display
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)

            # Forward pass
            with torch.no_grad():
                embeddings = encoder(inputs)  # Freeze encoder, get embeddings
            reconstructed = decoder(embeddings)

            # Compute loss
            loss = criterion(reconstructed, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{epoch_loss / len(dataloader):.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Generate and save reconstructed images
def save_reconstructed_images(encoder, decoder, dataloader, output_dir, device):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    # Wrap the dataloader with tqdm for progress display
    progress_bar = tqdm(dataloader, desc="Saving Reconstructed Images", unit="batch")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(device)
            embeddings = encoder(inputs)
            reconstructed = decoder(embeddings)

            # Rescale reconstructed images to [0, 1]
            reconstructed = (reconstructed + 1) / 2.0

            for i in range(reconstructed.size(0)):
                save_path = os.path.join(output_dir, f"reconstructed_{batch_idx * dataloader.batch_size + i + 1}.png")
                transforms.ToPILImage()(reconstructed[i].cpu()).save(save_path)

# Main execution
if __name__ == "__main__":
    # Initialize encoder and decoder
    encoder = EncoderModel(encoder_weights)
    decoder = DecoderModel()

    # Prepare CelebA data loader
    train_loader = get_celeba_dataloader(batch_size=32, train=True)
    test_loader = get_celeba_dataloader(batch_size=32, train=False)

    # Set up training components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    num_epochs = 10

    # Train the autoencoder
    print("Training the autoencoder...")
    train_autoencoder(encoder, decoder, train_loader, optimizer, criterion, num_epochs, device)

    # Save reconstructed images
    print("Saving reconstructed images...")
    save_reconstructed_images(encoder, decoder, test_loader, output_dir, device)
