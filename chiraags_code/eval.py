import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_test_data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import LogisticRegressionModel, NeuralNetwork


dataset_name = "cifar10"  # "mnist", "fashion_mnist", "cifar10"
model_types = ["logistic_regression", "nn"]
optimizers = ["Adadelta", "Adam", "Adamax", "LBFGS"]
seeds = [42, 43, 44]                             

results_dir = "./results"
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if dataset_name == "mnist" or dataset_name == "fashion_mnist":
    normalization_stats = (0.1307 * 255., 0.3081 * 255.) 
elif dataset_name == "cifar10":
    normalization_stats = (125.3, 63.0)  
else:
    raise ValueError(f"No normalization stats defined for {dataset_name}")


test_dataset = load_test_data(dataset_name, normalization_stats)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Get correct model
def get_model(model_type, input_dim, num_classes=2):
    if model_type == "logistic_regression":
        return LogisticRegressionModel(input_dim, num_classes)
    elif model_type == "nn":
        return NeuralNetwork(input_dim, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


sample_x, sample_y = next(iter(test_loader))
input_dim = sample_x.shape[1]
num_classes = len(torch.unique(sample_y))


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


results = {}

for model_type in model_types:
    for optimizer_name in optimizers:
        accuracies = []
        for seed in seeds:
            # Construct filename pattern
            pattern = f"{model_type}-{dataset_name}-{optimizer_name}-seed{seed}.pt"
            model_path = os.path.join(results_dir, pattern)

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}")
                continue

            # Load model
            model = get_model(model_type, input_dim, num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            # Evaluate
            acc = evaluate_model(model, test_loader, device)
            accuracies.append(acc)

        if len(accuracies) > 0:
            results[(model_type, optimizer_name)] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "count": len(accuracies)
            }


for model_type in model_types:
    # Filter results for this model_type
    model_results = {opt: r for ((mt, opt), r) in results.items() if mt == model_type}

    if not model_results:
        print(f"No results to plot for model_type: {model_type}")
        continue

    # Prepare data for plotting
    opts = list(model_results.keys())
    means = [model_results[o]["mean"] for o in opts]
    stds = [model_results[o]["std"] for o in opts]

    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(opts))

    plt.bar(x_positions, means, yerr=stds, capsize=5, alpha=0.7)
    plt.xticks(x_positions, opts)
    plt.ylabel("Test Accuracy")
    plt.title(f"{model_type} on {dataset_name} - Mean ± Std (n={len(seeds)})")
    plt.ylim([0, 1])

    # Save plot
    plot_path = os.path.join(results_dir, f"{model_type}-{dataset_name}-accuracy.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot to {plot_path}")
