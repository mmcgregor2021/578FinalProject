import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import time

train_data = dsets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = dsets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())

#Make data iterable
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

input_dim = 28*28
num_classes = 10
optimizers = ["Adam","Adamax","Adadelta","RMSprop", "L-BFGS"]

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

def train_and_test(optimizer_name, model, train_loader, test_loader, num_epochs):
    # lr = 0.1
    if optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=0.01)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    elif optimizer_name == "Adadelta":
        optimizer = optim.Adadelta(model.parameters())
    elif optimizer_name == "L-BFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    test_accuracies = []

    start_time = time.perf_counter()
    model.train()

    for epoch in range(num_epochs):
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
        losses.append(running_loss / len(train_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')


        iter_test = 0
        correct = 0
        total = 0
        model.eval()
        for images, labels in test_loader:
            iter_test += 1
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        test_accuracies.append(accuracy)
        print(f'Test Accuracy: {accuracy:.2f}%')

    end_time = time.perf_counter()
    accuracy = correct / total
    time_to_converge = end_time - start_time
    return accuracy, time_to_converge, losses, test_accuracies

def run_experiment(seed):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_iters = 6000

    accuracies = {}
    convergence_times = {}
    losses = {}
    test_accuracies = {}
    for opt_name in optimizers:
        model = LogisticRegressionModel(input_dim, num_classes)
        if opt_name == "L-BFGS":
            n_iters = 6000
        num_epochs = int(n_iters / (len(train_data) / batch_size))
        accuracies[opt_name], convergence_times[opt_name], losses[opt_name], test_accuracies[opt_name] = train_and_test(opt_name, model, train_loader, test_loader, num_epochs)

    # Plot loss curves
    plt.figure()
    for opt_name, loss_vals in losses.items():
        plt.plot(loss_vals, label=opt_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs Epoch for Different Optimizers (Seed {seed})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot test accuracies
    plt.figure()
    for opt_name, test_acc_vals in test_accuracies.items():
        plt.plot(test_acc_vals, label=opt_name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title(f'Test Accuracy vs Epoch for Different Optimizers (Seed {seed})')
    plt.legend()
    plt.grid(True)
    plt.show()


    return accuracies, convergence_times, losses


# Run experiments with different seeds
accuracies_by_seed = []
convergence_times_by_seed = []
losses_by_seed = []
seeds = [42, 43, 44]
# seeds = [42]
for seed in seeds:
    seed_results = run_experiment(seed)
    accuracies_by_seed.append(seed_results[0])
    convergence_times_by_seed.append(seed_results[1])
    losses_by_seed.append(seed_results[2])

# Calculate averages across seeds
avg_accuracies = {}
avg_convergence_times = {}
std_accuracies = {}
std_convergence_times = {}

for opt_name in optimizers:
    # Get values for each optimizer across seeds
    acc_values = [accuracies[opt_name] for accuracies in accuracies_by_seed]
    time_values = [convergence_times[opt_name] for convergence_times in convergence_times_by_seed]
    
    # Calculate means and standard deviations
    avg_accuracies[opt_name] = np.mean(acc_values)
    avg_convergence_times[opt_name] = np.mean(time_values)
    std_accuracies[opt_name] = np.std(acc_values)
    std_convergence_times[opt_name] = np.std(time_values)

# Plot averaged results with error bars
fig, ax1 = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(optimizers))

print(avg_accuracies.values())

# Plot accuracy bars with error bars
ax1.bar(index - bar_width/2, list(avg_accuracies.values()), bar_width, 
        yerr=list(std_accuracies.values()),
        label='Average Accuracy', color='blue', capsize=5)
ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Plot convergence time bars with error bars
ax2 = ax1.twinx()
ax2.bar(index + bar_width/2, list(avg_convergence_times.values()), bar_width,
        yerr=list(std_convergence_times.values()),
        label='Average Convergence Time (s)', color='red', capsize=5)
ax2.set_ylabel('Convergence Time (s)')
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')

# Adjust y-axis limits to make room for legend
ax1.set_ylim(0, max(list(avg_accuracies.values()) + list(std_accuracies.values())) * 1.2)
ax2.set_ylim(0, max(list(avg_convergence_times.values()) + list(std_convergence_times.values())) * 1.2)

plt.title("Fashion MNIST dataset (Averaged over seeds)")
plt.xticks(index, optimizers)
fig.tight_layout()
plt.show()

