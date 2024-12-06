import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def load_train_data(batch_size=64, val_split=0.2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Split into train and validation
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Number of classes
    num_classes = len(full_train_dataset.classes)  # Access class information directly

    # Input size
    input_size = full_train_dataset[0][0].numel()  # Flatten the input shape

    return train_loader, val_loader, num_classes, input_size

def load_test_data(batch_size=64):
    """
    Load and preprocess the MNIST test dataset.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
