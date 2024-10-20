import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Transformations to apply to the data (normalization)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training and test data
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Dataloaders for batching
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 10)  # Output layer for 10 classes (digits 0-9)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input (28x28 images)
        x = torch.relu(self.fc1(x))  # First layer with ReLU activation
        x = torch.relu(self.fc2(x))  # Second layer with ReLU activation
        x = self.fc3(x)  # Output layer (no activation, logits)
        return x


# Instantiate the model
model = NeuralNetwork()

# Loss function
criterion = nn.CrossEntropyLoss()

# Define optimizers
optimizers = {
    'Adamax': optim.Adamax(model.parameters(), lr=0.01),
    'Adadelta': optim.Adadelta(model.parameters(), lr=0.01),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001)
}

def train_model(model, optimizer, train_loader, num_epochs=5):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

accuracies = {}
for opt_name, optimizer in optimizers.items():
    print(f'\nTraining with {opt_name} optimizer:')

    # Reinitialize the model for each optimizer
    model = NeuralNetwork()

    # Train the model
    train_model(model, optimizer, train_loader, num_epochs=5)

    # Evaluate the model
    accuracies[opt_name] = evaluate_model(model, test_loader)


plt.bar(list(accuracies.keys()), list(accuracies.values()), width=0.4)
plt.xlabel("Optimizers")
plt.ylabel("Accuracy")
plt.title("Accuracy of different optimizers on MNIST dataset using neural network")
plt.show()
