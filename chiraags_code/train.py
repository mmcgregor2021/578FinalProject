import os
import torch
import numpy as np
import time
from datasets import load_train_data, load_test_data
from models import NeuralNetwork, LogisticRegressionModel
from torch.utils.data import DataLoader
from torch.optim import Adam, Adamax, Adadelta, LBFGS
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt


# Hyperparameters
SEEDS = [42, 43, 44]
DATASET_NAME = "mnist"  # "mnist", "fashion_mnist", "cifar10"
# OPTIMIZER_TYPES = ["Adadelta", "Adam", "Adamax"]
OPTIMIZER_TYPES = ["LBFGS"]

MODEL_TYPES = ["logreg"]  # nn, logreg # Currently only supporting "nn"
NUM_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.005
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_validate(model, train_loader, val_loader, test_loader, optimizer, loss_fn, optimizer_type, save_path):
    best_val_acc = 0
    epochs_no_improve = 0
    start_time = time.perf_counter()
    best_model_state = model.state_dict()

    train_losses = []  # List to store training losses per epoch
    val_losses = []    # List to store validation losses per epoch
    val_accuracies = []  # List to store validation accuracies per epoch

    for epoch in range(NUM_EPOCHS):
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            if optimizer_type == "LBFGS":
                outputs = None  # Initialize outputs variable

                def closure():
                    nonlocal outputs  # Allow closure to modify outputs
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))  # Track average training loss

        # Validate the model
        val_loss, val_acc = validate_and_get_metrics(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Acc: {100 * correct / total:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Check for improvement in validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

    # Save the best model
    torch.save(best_model_state, save_path)
    end_time = time.perf_counter()

    # Return collected metrics
    train_time = end_time - start_time
    return train_losses, val_losses, val_accuracies, train_time


def validate_and_get_metrics(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)  # Average validation loss
    val_acc = 100 * correct / total  # Validation accuracy in percentage
    return val_loss, val_acc

# Align arrays for averaging by padding shorter arrays with NaNs
def pad_and_average(losses):
    max_length = max(len(seed_losses) for seed_losses in losses)  # Find the longest array
    padded_losses = np.full((len(losses), max_length), np.nan)  # Initialize with NaNs

    for i, seed_losses in enumerate(losses):
        padded_losses[i, :len(seed_losses)] = seed_losses  # Fill with seed losses

    mean_losses = np.nanmean(padded_losses, axis=0)  # Compute mean ignoring NaNs
    return mean_losses

def main():
    all_train_losses = {optimizer_type: [] for optimizer_type in OPTIMIZER_TYPES}
    all_val_losses = {optimizer_type: [] for optimizer_type in OPTIMIZER_TYPES}
    all_val_accuracies = {optimizer_type: [] for optimizer_type in OPTIMIZER_TYPES}

    for seed in SEEDS:
        set_seed(seed)
        train_loader, val_loader, num_classes, input_size = load_train_data(batch_size=BATCH_SIZE)
        test_loader = load_test_data(batch_size=BATCH_SIZE)

        for model_type in MODEL_TYPES:
            for optimizer_type in OPTIMIZER_TYPES:
                print(f"\nTraining {model_type} model with {optimizer_type} optimizer (Seed {seed})")

                if model_type == "nn":
                    model = NeuralNetwork(input_size, num_classes)
                elif model_type == "logreg":
                    model = LogisticRegressionModel(input_size, num_classes)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

                save_path = os.path.join(SAVE_DIR, f"{model_type}-{DATASET_NAME}-{optimizer_type}-seed{seed}.pt")

                if optimizer_type == "Adam":
                    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
                elif optimizer_type == "Adamax":
                    optimizer = Adamax(model.parameters(), lr=LEARNING_RATE)
                elif optimizer_type == "Adadelta":
                    optimizer = Adadelta(model.parameters())
                elif optimizer_type == "LBFGS":
                    optimizer = LBFGS(model.parameters(), lr=0.001)
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer_type}")

                loss_fn = CrossEntropyLoss()
                train_losses, val_losses, val_accuracies, train_time = train_and_validate(
                    model, train_loader, val_loader, test_loader, optimizer, loss_fn, optimizer_type, save_path
                )
                print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%, Training Time: {train_time:.2f}s")

                # Accumulate losses and accuracies for plotting
                all_train_losses[optimizer_type].append(train_losses)
                all_val_losses[optimizer_type].append(val_losses)
                all_val_accuracies[optimizer_type].append(val_accuracies)

    # Plot and save validation loss curves
    plt.figure()

    for optimizer_type, losses in all_val_losses.items():
        mean_losses = pad_and_average(losses)  # Average across seeds
        plt.plot(range(1, len(mean_losses) + 1), mean_losses, label=optimizer_type)
    
    plt.title(f'Validation Loss for {model_type} on {DATASET_NAME}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f'{model_type}_{DATASET_NAME}_validation_loss.png'))
    plt.close()

    # Plot and save validation accuracy curves
    plt.figure()
    for optimizer_type, accuracies in all_val_accuracies.items():
        mean_accuracies = pad_and_average(accuracies)  # Average across seeds
        plt.plot(range(1, len(mean_accuracies) + 1), mean_accuracies, label=optimizer_type)
    
    plt.title(f'Validation Accuracy for {model_type} on {DATASET_NAME}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f'{model_type}_{DATASET_NAME}_validation_accuracy.png'))
    plt.close()

if __name__ == "__main__":
    main()