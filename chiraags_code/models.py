import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for 10 classes (digits 0-9)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input (28x28 images)
        x = torch.relu(self.fc1(x))  # First layer with ReLU activation
        x = torch.relu(self.fc2(x))  # Second layer with ReLU activation
        x = self.fc3(x)  # Output layer (no activation, logits)
        return x