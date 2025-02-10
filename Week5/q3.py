import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=50, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=50, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model's state dict: ")
for param_tensor in model.state_dict().keys():
    print(param_tensor,"\t", model.state_dict()[param_tensor].size())
print()

print("Optimizer's state dict: ")
for var_name in optimizer.state_dict():
    print(var_name,"\t", optimizer.state_dict()[var_name])
print()

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    epoch_accuracy = (correct / total) * 100
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {epoch_accuracy:.2f}%')

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

correct = np.sum(np.array(all_preds) == np.array(all_labels))
total = len(all_labels)
accuracy = correct/total
print(f'Accuracy: {accuracy * 100:.4f}%')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f'Number of learnable parameters: {num_params}')

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

torch.save(model,"./ModelFiles/mnist_model.pt")

# Output :
#     Model's state dict:
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
#     Optimizer's state dict:
#     state 	 {}
#     param_groups 	 [{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
#
#     Epoch [1/5], Loss: 0.26823175470422334, Accuracy: 91.55%
#     Epoch [2/5], Loss: 0.084577521016278, Accuracy: 97.42%
#     Epoch [3/5], Loss: 0.061520795834221646, Accuracy: 98.11%
#     Epoch [4/5], Loss: 0.048769522910006344, Accuracy: 98.47%
#     Epoch [5/5], Loss: 0.03942956825097402, Accuracy: 98.76%
#     Accuracy: 98.1400%
#     Number of learnable parameters: 149798
