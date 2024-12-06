import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import time
from torch.utils.data import random_split
import os


seeds = [42, 43, 44, 45, 46]
total_accuracies = []
total_convergence = []
TOTAL_EPOCHS = []
optimizers = ["Adadelta", "Adam", "Adamax", "L-BFGS"]
results = {opt: [] for opt in optimizers}
all_val_accuracies = {opt: [] for opt in optimizers}
only_test = True
PATH = "/Users/brunajasinowodolinski/Desktop/578FinalProject/brunas_code"


for seed in seeds:
    # Set the seed for reproducibility
    lr = 0.005
    num_epochs = 50
    final_epoch = 0

    patience = 5  # Number of epochs to wait for improvement
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = dsets.MNIST(root='./data', train=False, transform=transform)

    # train_data = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # test_data = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # Make data iterable
    batch_size = 100 # was 64
    val_split = 0.2
    # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LogisticRegressionModel, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            return self.fc(x)


    def validate(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total


    def train(optimizer_name, model, train_loader):
        if optimizer_name == "Adamax":
            optimizer = optim.Adamax(model.parameters(), lr=lr)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_name == "Adadelta":
            optimizer = optim.Adadelta(model.parameters())
        elif optimizer_name == "L-BFGS":
            optimizer = optim.LBFGS(
                model.parameters(),
                lr=0.001,
                max_iter=5,
                max_eval=8,
                tolerance_grad=1e-5,
                tolerance_change=1e-9,
                history_size=20,
                line_search_fn="strong_wolfe"
            )
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        epochs_no_improve = 0
        start_time = time.perf_counter()
        model.train()
        val_accs = []


        for epoch in range(num_epochs):
            final_epoch = epoch
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                TOTAL_EPOCHS.append(final_epoch)
                
                break

            running_loss = 0
            total = 0
            correct = 0
            for i, (images, labels) in enumerate(train_loader):
                if optimizer_name == "L-BFGS":
                    def closure():
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)
                        loss.backward()
                        return loss

                    loss = optimizer.step(closure)
                    outputs = model(images)
                else:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            val_acc = validate(model, val_loader) # changed to the val loader

            val_accs.append(val_acc)


            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
                f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            

        end_time = time.perf_counter()
        final_val_acc = best_val_acc
        time_to_converge = end_time - start_time
        save_path = os.path.join(f"{PATH}/data", f"logreg--{optimizer_name}-seed{seed}.pt")

        torch.save(best_model_state, save_path)
        all_val_accuracies[optimizer_name].append(val_accs)

    
        return final_val_acc, time_to_converge
    
        
    def test(optimizer_name, model, test_loader):
        """
        Load the best model state saved during training and evaluate its performance on the test set.

        Args:
            optimizer_name (str): The name of the optimizer used.
            model (nn.Module): The model architecture.
            test_loader (DataLoader): DataLoader for the test dataset.

        Returns:
            float: Test accuracy of the model.
        """
        # Load the saved model state
        model_path = os.path.join(f"{PATH}/data", f"logreg--{optimizer_name}-seed{seed}.pt")
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        print(f"Test Accuracy with {optimizer_name}: {test_accuracy:.2f}%")
        return test_accuracy


    input_dim = 28 * 28
    num_classes = 10

    accuracies = {}
    convergence_times = {}
    for opt_name in optimizers:
        model = LogisticRegressionModel(input_dim, num_classes)
        if not only_test: 
            _, convergence_times[opt_name] = train(opt_name, model, train_loader)
        accuracies[opt_name] = test(opt_name, model, test_loader)

        results[opt_name].append(accuracies[opt_name])

    print("Final Accuracies:", accuracies)
    print("Convergence Times:", convergence_times)



print(results)
print(TOTAL_EPOCHS)

optimizers = list(results.keys())
means = [np.mean(results[opt]) for opt in optimizers]
stds = [np.std(results[opt]) for opt in optimizers]
 
plt.figure(figsize=(10, 6))
x_positions = np.arange(len(optimizers))

# Dynamically adjust ylim to focus on the range of accuracies
min_y = max(0, min(means) - 0.05)  # Minimum accuracy minus padding
max_y = min(1, max(means) + 0.05)  # Maximum accuracy plus padding

plt.bar(x_positions, means, yerr=stds, capsize=5, alpha=0.7)
plt.xticks(x_positions, optimizers)
plt.ylabel("Test Accuracy")
plt.title(f"Logistic Regression on MNIST - Mean ± Std (n={len(seeds)})")
plt.ylim([min_y, max_y])  # Set the dynamic range

plot_path = os.path.join(f"{PATH}/results", f"logreg_accuracy.png")
plt.savefig(plot_path)
plt.close()

if not only_test: 
    with open("logreg_val_accuracies.txt", "a") as file:
        file.write(f"{all_val_accuracies}")



