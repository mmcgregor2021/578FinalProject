import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from sklearn.model_selection import train_test_split

# Normalization constants
NORMALIZATION_CONSTANTS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
}

def load_train_data(dataset_name, validation_split=0.2):
    """
    Load and preprocess the training dataset. Optionally split into train/validation sets.
    
    Args:
        dataset_name (str): Name of the dataset to load. Must be one of 'mnist', 'fashion_mnist', or 'cifar10'.
        validation_split (float): Fraction of the training data to use as validation.

    Returns:
        train_dataset (TensorDataset): Preprocessed training dataset.
        val_dataset (TensorDataset): Preprocessed validation dataset.
        normalization_stats (tuple): Mean and standard deviation used for normalization.
    """
    if dataset_name == "mnist":
        dataset = MNIST(root='./data', train=True, download=True)
    elif dataset_name == "fashion_mnist":
        dataset = FashionMNIST(root='./data', train=True, download=True)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(root='./data', train=True, download=True)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    # Extract data and labels
    data = dataset.data
    labels = dataset.targets

    if dataset_name == "cifar10":
        # Convert to binary classification
        data = np.array(data)
        labels = np.array(labels)
        idx = np.where((labels == 0) | (labels == 1))[0]
        data = data[idx]
        labels = labels[idx]

    else:
        data = data.numpy()
        labels = labels.numpy()

    # Reshape data
    data = data.reshape(data.shape[0], -1).astype(np.float32)

    # Compute normalization statistics from the training set
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-7  # Avoid divide-by-zero errors
    normalization_stats = (mean, std)

    # Normalize the training data
    data = (data - mean) / std

    # Split into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=validation_split, random_state=42, stratify=labels
    )

    # Convert to TensorDatasets
    train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels))
    val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.LongTensor(val_labels))

    return train_dataset, val_dataset, normalization_stats

def load_test_data(dataset_name, normalization_stats):

    if dataset_name == "mnist":
        dataset = MNIST(root='./data', train=False, download=True)
    elif dataset_name == "fashion_mnist":
        dataset = FashionMNIST(root='./data', train=False, download=True)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(root='./data', train=False, download=True)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    # Extract data and labels
    data = dataset.data
    labels = dataset.targets

    if dataset_name == "cifar10":
        # Convert to binary classification
        data = np.array(data)
        labels = np.array(labels)
        idx = np.where((labels == 0) | (labels == 1))[0]
        data = data[idx]
        labels = labels[idx]


    else:
        data = data.numpy()
        labels = labels.numpy()

    # Reshape data
    data = data.reshape(data.shape[0], -1).astype(np.float32)

    # Normalize the test data using training set statistics
    mean, std = normalization_stats
    data = (data - mean) / std

    # Convert to TensorDataset
    test_dataset = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))

    return test_dataset
