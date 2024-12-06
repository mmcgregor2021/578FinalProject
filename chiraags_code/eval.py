import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_test_data
from torch.utils.data import DataLoader
from models import NeuralNetwork, LogisticRegressionModel

# Parameters
dataset_name = "mnist"  # "mnist", "fashion_mnist", "cifar10"
model_types = ["logreg"]  # nn, logreg
optimizers = ["Adadelta", "Adam", "Adamax", "LBFGS"]
seeds = [42, 43, 44]
results_dir = "./results"
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
test_loader = load_test_data(batch_size=batch_size)

# Function to load the model
def get_nn_model(input_dim, num_classes):
    return NeuralNetwork(input_dim, num_classes)

def get_logreg_model(input_dim, num_classes):
    return LogisticRegressionModel(input_dim, num_classes)

# Function to evaluate the model
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# Extract data properties
sample_x, sample_y = next(iter(test_loader))
input_dim = sample_x.view(sample_x.size(0), -1).shape[1]
num_classes = len(torch.unique(sample_y))

results = {}

# Evaluate all combinations of model types and optimizers
for model_type in model_types:
    for optimizer_name in optimizers:
        accuracies = []
        for seed in seeds:
            model_filename = f"{model_type}-{dataset_name}-{optimizer_name}-seed{seed}.pt"
            model_path = os.path.join(results_dir, model_filename)

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}")
                continue

            # Load model
            if model_type == "nn":
                model = get_nn_model(input_dim, num_classes)
            elif model_type == "logreg":
                model = get_logreg_model(input_dim,num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            # Evaluate model
            accuracy = evaluate_model(model, test_loader, device)
            accuracies.append(accuracy)

        if len(accuracies) > 0:
            print(accuracies)
            print(optimizer_name)
            results[(model_type, optimizer_name)] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "count": len(accuracies),
            }

# Plot results for each model type
for model_type in model_types:
    model_results = {opt: r for ((mt, opt), r) in results.items() if mt == model_type}

    if not model_results:
        print(f"No results to plot for model_type: {model_type}")
        continue

    optimizers = list(model_results.keys())
    means = [model_results[opt]["mean"] for opt in optimizers]
    stds = [model_results[opt]["std"] for opt in optimizers]

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(optimizers))

    # Dynamically adjust ylim to focus on the range of accuracies
    min_y = max(0, min(means) - 0.05)  # Minimum accuracy minus padding
    max_y = min(1, max(means) + 0.05)  # Maximum accuracy plus padding

    plt.bar(x_positions, means, yerr=stds, capsize=5, alpha=0.7)
    plt.xticks(x_positions, optimizers)
    plt.ylabel("Test Accuracy")
    plt.title(f"{model_type} on {dataset_name} - Mean ± Std (n={len(seeds)})")
    plt.ylim([min_y, max_y])  # Set the dynamic range

    plot_path = os.path.join(results_dir, f"{model_type}-{dataset_name}-accuracy.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to {plot_path}")
