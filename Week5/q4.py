import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Modified CNN Classifier that accepts filter sizes dynamically
class CNNClassifierReduced(nn.Module):
    def __init__(self, filter_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, filter_size, kernel_size=3),  # dynamic filter size
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(filter_size, filter_size * 2, kernel_size=3),  # dynamic filter size
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(filter_size * 2, filter_size, kernel_size=3),  # dynamic filter size
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self._to_linear = None
        self._calculate_conv_output_size((1, 28, 28))

        self.classification_head = nn.Sequential(
            nn.Linear(self._to_linear, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True)
        )

    def _calculate_conv_output_size(self, input_size):
        with torch.no_grad():
            x = torch.ones(1, *input_size)
            x = self.net(x)
            self._to_linear = x.numel()

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))


# Load the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=50, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=50, shuffle=False)


# Function to count learnable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Function to train and evaluate the model
def train_and_evaluate(model, train_loader, test_loader, num_epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    accuracy = correct / total
    return accuracy


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment to track accuracy and parameter changes
param_drop_percentage = []
accuracies = []

# Create the model and get the number of parameters for the original model
original_filter_size = 32
original_model = CNNClassifierReduced(filter_size=original_filter_size)
initial_num_params = count_parameters(original_model)

# Define the filter sizes to experiment with
filter_sizes = [16, 32, 64]  # Smaller, default, and larger filter sizes

for filter_size in filter_sizes:
    model = CNNClassifierReduced(filter_size=filter_size)
    num_params = count_parameters(model)
    param_drop_percentage.append(100 * (initial_num_params - num_params) / initial_num_params)

    accuracy = train_and_evaluate(model, train_loader, test_loader, num_epochs=5)
    accuracies.append(accuracy)

# Plot percentage drop in parameters vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(param_drop_percentage, accuracies, marker='o')
plt.xlabel('Percentage Drop in Parameters')
plt.ylabel('Accuracy')
plt.title('Percentage Drop in Parameters vs Accuracy')
plt.grid(True)
plt.show()
