import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

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
        if i % 100 == 99:
            print(f'[{epoch+1},{i+1}] loss: {loss.item()}')
            running_loss = 0.0

print(f"\nFinal loss = {loss.item()}, Total params = {total_params}")

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

print(f"\nCorrect = {correct}, Total = {total}")

print("\nConfusion matrix : ")
for i in range(10):
    print(mat[i])

