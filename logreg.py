import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

#Load the data
train_data = dsets.MNIST(root = './data', train=True, download=True, transform=transforms.ToTensor())
test_data = dsets.MNIST(root = './data', train=False, transform = transforms.ToTensor())



#Make data iterable
batch_size = 100
n_iters = 3000
num_epochs = int(n_iters / (len(train_data) / batch_size))
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def train_model(model, optimizer, num_epochs, train_loader):
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.view(-1, 28*28).requires_grad_()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

def test_model(model, test_loader):
    iter_test = 0
    correct = 0
    total = 0
    model.eval()
    for images, labels in test_loader:
        iter_test += 1
        images = images.view(-1, 28 * 28).requires_grad_()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return (correct / total)


loss_func = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
input_dim = 28*28
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)
optimizers = {"Adam":torch.optim.Adam(model.parameters(), lr=learning_rate),
              "Adamax":torch.optim.Adamax(model.parameters(), lr=learning_rate),
              "Adadelta":torch.optim.Adadelta(model.parameters(), lr=learning_rate),
              "RMSprop":torch.optim.RMSprop(model.parameters(),lr=learning_rate)}
accuracies = {}
for opt_name, optimizer in optimizers.items():
    model = LogisticRegressionModel(28*28, 10)
    train_model(model, optimizer, num_epochs, train_loader)
    accuracies[opt_name] = test_model(model, test_loader)





plt.bar(list(accuracies.keys()), list(accuracies.values()), width=0.4)
plt.xlabel("Optimizers")
plt.ylabel("Accuracy")
plt.title("Accuracy of different opitmizers on MNIST dataset using logistic regression")
plt.show()
