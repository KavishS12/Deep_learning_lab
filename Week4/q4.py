import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.net = nn.Sequential(nn.Linear(784, 100, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(100, 100, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(100, 10, bias=True))

    def forward(self, x):
        x = x.view(-1, 784)
        return self.net(x)


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=50, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=50, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

total_params = 0
for name, param in model.named_parameters():
    params = param.numel()
    total_params += params

for epoch in range(6):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{6}], Loss: {running_loss / len(train_loader)}')

print(f"Final loss = {loss.item()}, Total params = {total_params}")

mat = [[0 for _ in range(10)] for _ in range(10)]
correct, total = 0, 0
for i, vdata in enumerate(test_loader):
    tinputs, tlabels = vdata[0].to(device), vdata[1].to(device)
    toutputs = model(tinputs)

    _, predicted = torch.max(toutputs, 1)
    total += tlabels.size(0)
    correct += (predicted == tlabels).sum()
    for i in range(len(predicted)):
        mat[predicted[i].item()][tlabels[i].item()] += 1

print(f"Correct = {correct}, Total = {total}, Accuracy = {correct/total:.2f}")

plt.figure(figsize=(8,6))
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Output :
    # Epoch [1/6], Loss: 1.498233386153976
    # Epoch [2/6], Loss: 0.4762321992591023
    # Epoch [3/6], Loss: 0.3593600119339923
    # Epoch [4/6], Loss: 0.3159635362277428
    # Epoch [5/6], Loss: 0.2887742422365894
    # Epoch [6/6], Loss: 0.26687682832591236
    # Final loss = 0.29278454184532166, Total params = 89610
    # Correct = 9303, Total = 10000, Accuracy = 0.93