import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
optimizers = ["Adam","Adamax","Adadelta","RMSprop", "L-BFGS"]
# optimizers = ["L-BFGS"]

def train_and_test(model, optimizer_name, train_loader, test_loader, num_epochs=10):
    if optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=0.01)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    elif optimizer_name == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    elif optimizer_name == "L-BFGS":
        optimizer = optim.LBFGS(model.parameters(), 
                               lr=0.01,
                               max_iter=20,
                               history_size=10,
                               line_search_fn="strong_wolfe")
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            # For L-BFGS, we need to define a closure function
            if optimizer_name == "L-BFGS":
                def closure():
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss
                
                # Move these outside the closure
                optimizer.zero_grad()
                loss = optimizer.step(closure)
                outputs = model(images)
                
                # Use the actual loss value
                running_loss += loss.item()
            else:
                # Original code for other optimizers
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.perf_counter()
    accuracy = correct / total
    convergence_time = end_time-start_time
    return accuracy, convergence_time

accuracies = {}
convergence_times = {}
for opt_name in optimizers:
    print(f'\nTraining with {opt_name} optimizer:')

    # Reinitialize the model for each optimizer
    model = NeuralNetwork()

    # Train the model
    accuracies[opt_name], convergence_times[opt_name] = train_and_test(model, opt_name, train_loader, test_loader)


# Plotting the results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for accuracy and convergence time
bar_width = 0.35  # Width of the bars
index = np.arange(len(list(convergence_times.keys())))  # x positions for each optimizer

# Bar plot for accuracy (shifted by -bar_width/2)
ax1.bar(index - bar_width/2, list(accuracies.values()), bar_width, label='Accuracy', color='blue')
ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.75, 1)
ax1.tick_params(axis='y')
ax1.legend(loc = 'upper left')

# Bar plot for convergence time (shifted by bar_width/2)
ax2 = ax1.twinx()
ax2.bar(index + bar_width/2, list(convergence_times.values()), bar_width, label='Convergence Time (s)', color='red')
ax2.set_ylabel('Convergence Time (s)')
#ax2.set_ylim(14, 18)
ax2.tick_params(axis='y')
ax2.legend(loc = 'upper right')

# Labels and title
plt.title("MNIST dataset (NN)")
plt.xticks(index, list(convergence_times.keys()))
fig.tight_layout()
plt.show()

