import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import time

#Load the data
#train_data = dsets.MNIST(root = './data', train=True, download=True, transform=transforms.ToTensor())
#test_data = dsets.MNIST(root = './data', train=False, transform = transforms.ToTensor())

train_data = dsets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = dsets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())


#Make data iterable
batch_size = 100
n_iters = 3000
num_epochs = int(n_iters / (len(train_data) / batch_size))
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)



def train_and_test(optimizer_name, model, train_loader, test_loader):
    lr = 0.1
    if optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "Adadelta":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "L-BFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

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

    end_time = time.perf_counter()
    accuracy = correct / total
    time_to_converge = end_time - start_time
    return accuracy, time_to_converge

input_dim = 28*28
num_classes = 10
#model = LogisticRegressionModel(input_dim, num_classes)
optimizers = ["Adam","Adamax","Adadelta","RMSprop", "L-BFGS"]
accuracies = {}
convergence_times = {}
for opt_name in optimizers:
    model = LogisticRegressionModel(input_dim, num_classes)
    accuracies[opt_name], convergence_times[opt_name] = train_and_test(opt_name, model, train_loader, test_loader)

# Plotting the results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for accuracy and convergence time
bar_width = 0.35  # Width of the bars
index = np.arange(len(list(convergence_times.keys())))  # x positions for each optimizer

# Bar plot for accuracy (shifted by -bar_width/2)
ax1.bar(index - bar_width/2, list(accuracies.values()), bar_width, label='Accuracy', color='blue')
ax1.set_xlabel('Optimizer')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.7, .9)
ax1.tick_params(axis='y')
ax1.legend(loc = 'upper left')

# Bar plot for convergence time (shifted by bar_width/2)
ax2 = ax1.twinx()
ax2.bar(index + bar_width/2, list(convergence_times.values()), bar_width, label='Convergence Time (s)', color='red')
ax2.set_ylabel('Convergence Time (s)')
ax2.set_ylim(14, 18)
ax2.tick_params(axis='y')
ax2.legend(loc = 'upper right')

# Labels and title
plt.title("Fashion MNIST dataset with learning rate 0.1")
plt.xticks(index, list(convergence_times.keys()))
fig.tight_layout()
plt.show()






