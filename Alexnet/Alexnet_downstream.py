import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from ijepa_alexnet_training_v2 import IJEPA_Model

# Define the new model
class NewModel(nn.Module):
    def __init__(self, encoder):
        super(NewModel, self).__init__()
        self.encoder = encoder  # Reuse the pre-trained encoder
        self.new_forward_layer = nn.Sequential(
            nn.Linear(512, 256),  # Adjust input and output dimensions as needed
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 classes for CIFAR-10
        )

    def forward(self, x):
        with torch.no_grad():  # Freeze encoder if required
            x = self.encoder(x)
        x = self.new_forward_layer(x)
        return x

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Load the pre-trained model and extract the encoder
checkpoint_path = "ijeepa_model.pth"
ijeepa_model = IJEPA_Model()
checkpoint = torch.load(checkpoint_path)
ijeepa_model.load_state_dict(checkpoint['model_state_dict'])
encoder = ijeepa_model.encoder

# Initialize the new model
new_model = NewModel(encoder)
for param in new_model.encoder.parameters():
    param.requires_grad = False  # Freeze encoder weights
new_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(new_model.new_forward_layer.parameters(), lr=0.001)

# Train the new model
def train_and_evaluate(model, trainloader, testloader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)

        # Evaluate on the test set
        test_accuracy = evaluate(model, testloader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_accuracies

# Evaluation function
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Train and plot results
epochs = 10
train_losses, test_accuracies = train_and_evaluate(new_model, trainloader, testloader, epochs)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
