import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from tqdm import tqdm  # Progress bar

# Load the checkpoint
checkpoint_path = '/scratch/gilbreth/abelde/DL_Research/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar'
output_dir = '/scratch/gilbreth/abelde/DL_Research/results'

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

# Define the classification model
class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(1280, num_classes)  # 1280 matches ViT output

    def forward(self, x):
        features = self.encoder(x)  # Extract features
        logits = self.classifier(features)  # Classify features
        return logits

# Prepare CIFAR-100 dataset
def get_cifar100_dataloader(batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)

# Train the model
def train_model(model, train_loader, optimizer, criterion, num_epochs, device, checkpoint_dir):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Wrap the dataloader with tqdm for progress display
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar with current loss and accuracy
            progress_bar.set_postfix(loss=f"{epoch_loss / len(train_loader):.4f}", acc=f"{100.0 * correct / total:.2f}%")

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {100.0 * correct / total:.2f}%")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"cifar100_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / len(train_loader),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Evaluate the model
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    accuracy_metric = Accuracy(task="multiclass", num_classes=100).to(device)

    # Wrap the dataloader with tqdm for progress display
    progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            accuracy_metric.update(predicted, labels)

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Metric-Based Test Accuracy: {accuracy_metric.compute().item() * 100:.2f}%")

# Main execution
if __name__ == "__main__":
    # Initialize encoder
    encoder = EncoderModel(encoder_weights)

    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    # Initialize classification model
    model = ClassificationModel(encoder, num_classes=100)

    # Prepare CIFAR-100 data loaders
    train_loader = get_cifar100_dataloader(batch_size=32, train=True)
    test_loader = get_cifar100_dataloader(batch_size=32, train=False)

    # Set up training components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Define checkpoint directory
    checkpoint_dir = output_dir  # Use the results directory for saving checkpoints

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, optimizer, criterion, num_epochs, device, checkpoint_dir)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_loader, device)
