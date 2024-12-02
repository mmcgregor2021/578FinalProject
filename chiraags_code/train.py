import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam, Adamax, Adadelta, LBFGS
from torch.nn import CrossEntropyLoss
from datasets import load_train_data, load_test_data
from models import LogisticRegressionModel, NeuralNetwork

# Hyperparameters
SEEDS = [42, 43, 44]
DATASET_NAME = "cifar10"  # Options: "mnist", "fashion_mnist", "cifar10"
MODEL_TYPES = ["logistic_regression", "nn"]
OPTIMIZER_TYPES = ["Adam", "Adamax", "Adadelta"] # Add LBFGS
BATCH_SIZE = 64
NUM_EPOCHS = 10
NUM_EPOCHS = 50  # High, so early stopping can be effective
PATIENCE = 5  # Number of epochs to wait for improvement
EARLY_STOPPING = True

# Directory to save model weights and logs
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def initialize_model(model_type, input_size, num_classes):

    if model_type == "logistic_regression":
        return LogisticRegressionModel(input_size, num_classes)
    elif model_type == "nn":
        return NeuralNetwork(input_size, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def initialize_optimizer(optimizer_type, model):

    if optimizer_type == "Adam":
        return Adam(model.parameters(), lr=0.01)
    elif optimizer_type == "Adamax":
        return Adamax(model.parameters(), lr=0.01)
    elif optimizer_type == "Adadelta":
        return Adadelta(model.parameters(), lr=1.0)
    elif optimizer_type == "LBFGS":
        return LBFGS(model.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, optimizer_type, model_type, early_stopping=False, patience=5, save_path=None):
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_accuracies = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            # For LBFGS optimizer, closure function is required
            if optimizer_type == "LBFGS":
                def closure():
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

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_accuracy = train_correct / train_total

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            # Save the best model
            if save_path:
                torch.save(best_model_state, save_path)
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Early stopping
        if early_stopping and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, val_accuracies, val_losses, best_model_state


def main():

    # Load datasets
    train_dataset, val_dataset, normalization_stats = load_train_data(DATASET_NAME)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = train_dataset[0][0].shape[0]
    num_classes = len(set(train_dataset[:][1].tolist()))  # Infer number of classes

    for model_type in MODEL_TYPES:
        # Initialize dictionaries to hold metrics for each optimizer
        all_train_losses = {}
        all_val_accuracies = {}
        all_val_losses = {}
        for optimizer_type in OPTIMIZER_TYPES:
            avg_train_losses = np.zeros(NUM_EPOCHS)
            avg_val_accuracies = np.zeros(NUM_EPOCHS)
            avg_val_losses = np.zeros(NUM_EPOCHS)
            actual_epochs = NUM_EPOCHS

            for seed in SEEDS:
                set_seed(seed)
                # Initialize model, optimizer, and loss function
                model = initialize_model(model_type, input_size, num_classes)
                optimizer = initialize_optimizer(optimizer_type, model)
                loss_fn = CrossEntropyLoss()

                print(f"\nTraining {model_type} with {optimizer_type} optimizer, Seed: {seed}")
                save_path = os.path.join(SAVE_DIR, f"{model_type}-{DATASET_NAME}-{optimizer_type}-seed{seed}.pt")
                
                # Run training and validation
                train_losses, val_accuracies, val_losses, best_model_state = train_and_validate(
                    model, train_loader, val_loader, optimizer, loss_fn, NUM_EPOCHS, optimizer_type, model_type,
                    early_stopping=EARLY_STOPPING, patience=PATIENCE, save_path=save_path
                )

                epochs_run = len(train_losses)
                avg_train_losses[:epochs_run] += np.array(train_losses)
                avg_val_accuracies[:epochs_run] += np.array(val_accuracies)
                avg_val_losses[:epochs_run] += np.array(val_losses)
                actual_epochs = min(actual_epochs, epochs_run)

            # Average the metrics over all seeds
            avg_train_losses = avg_train_losses[:actual_epochs] / len(SEEDS)
            avg_val_accuracies = avg_val_accuracies[:actual_epochs] / len(SEEDS)
            avg_val_losses = avg_val_losses[:actual_epochs] / len(SEEDS)

            all_train_losses[optimizer_type] = avg_train_losses
            all_val_accuracies[optimizer_type] = avg_val_accuracies
            all_val_losses[optimizer_type] = avg_val_losses

        # Plot and save the loss curves
        plt.figure()
        for optimizer_type in OPTIMIZER_TYPES:
            plt.plot(range(1, len(all_train_losses[optimizer_type]) + 1), all_train_losses[optimizer_type], label=optimizer_type)
        plt.title(f'Training Loss for {model_type} on {DATASET_NAME}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f'{model_type}_{DATASET_NAME}_training_loss.png'))
        plt.close()

        # Plot and save the validation loss curves
        plt.figure()
        for optimizer_type in OPTIMIZER_TYPES:
            plt.plot(range(1, len(all_val_losses[optimizer_type]) + 1), all_val_losses[optimizer_type], label=optimizer_type)
        plt.title(f'Validation Loss for {model_type} on {DATASET_NAME}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f'{model_type}_{DATASET_NAME}_validation_loss.png'))
        plt.close()

        # Plot and save the validation accuracy curves
        plt.figure()
        for optimizer_type in OPTIMIZER_TYPES:
            plt.plot(range(1, len(all_val_accuracies[optimizer_type]) + 1), all_val_accuracies[optimizer_type], label=optimizer_type)
        plt.title(f'Validation Accuracy for {model_type} on {DATASET_NAME}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f'{model_type}_{DATASET_NAME}_validation_accuracy.png'))
        plt.close()

if __name__ == "__main__":
    main()