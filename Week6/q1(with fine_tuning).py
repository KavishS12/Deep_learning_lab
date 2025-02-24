import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self._to_linear = None
        self._calculate_conv_output_size((1, 28, 28))

        self.classification_head = nn.Sequential(
            nn.Linear(self._to_linear, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True)
        )

    def _calculate_conv_output_size(self, input_size):
        with torch.no_grad():
            x = torch.ones(1, *input_size)
            x = self.net(x)
            self._to_linear = x.numel()

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))

fashion_mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(fashion_mnist_trainset, batch_size=50, shuffle=True)

fasion_mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(fasion_mnist_testset, batch_size=50, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("../Week5/ModelFiles/mnist_model.pt")
model.to(device)

print("Model's state_dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t",model.state_dict()[param_tensor].size())
print()

# Freeze all layers except the final classification head for fine-tuning
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final classification layers
for param in model.classification_head.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.classification_head.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
model.train()
epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    epoch_accuracy = (correct / total) * 100
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Final accuracy after fine-tuning: {accuracy:.4f}%")

# Output :
#     Model's state_dict:
#     net.0.weight 	 torch.Size([64, 1, 3, 3])
#     net.0.bias 	 torch.Size([64])
#     net.3.weight 	 torch.Size([128, 64, 3, 3])
#     net.3.bias 	 torch.Size([128])
#     net.6.weight 	 torch.Size([64, 128, 3, 3])
#     net.6.bias 	 torch.Size([64])
#     classification_head.0.weight 	 torch.Size([20, 64])
#     classification_head.0.bias 	 torch.Size([20])
#     classification_head.2.weight 	 torch.Size([10, 20])
#     classification_head.2.bias 	 torch.Size([10])
#
#     Epoch [1/10], Loss: 2.0936, Accuracy: 31.89%
#     Epoch [2/10], Loss: 1.1738, Accuracy: 59.81%
#     Epoch [3/10], Loss: 0.9712, Accuracy: 66.19%
#     Epoch [4/10], Loss: 0.8903, Accuracy: 68.65%
#     Epoch [5/10], Loss: 0.8461, Accuracy: 70.07%
#     Epoch [6/10], Loss: 0.8167, Accuracy: 71.10%
#     Epoch [7/10], Loss: 0.7953, Accuracy: 71.87%
#     Epoch [8/10], Loss: 0.7781, Accuracy: 72.54%
#     Epoch [9/10], Loss: 0.7617, Accuracy: 73.06%
#     Epoch [10/10], Loss: 0.7498, Accuracy: 73.39%
#     Final accuracy after fine-tuning: 73.1200%
