import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

class CustomDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.training:
            # Generate a mask using Bernoulli distribution
            mask = (torch.rand_like(x) > self.dropout_rate).float().to(x.device)
            # Scale the output to maintain expected value
            return x * mask / (1 - self.dropout_rate)
        else:
            return x

class CNNWithCustomDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNWithCustomDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = CustomDropout(dropout_rate)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply custom dropout
        x = self.fc2(x)
        return x

class CNNWithBuiltInDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNWithBuiltInDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

def evaluate_model(model, test_loader):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    acc = accuracy_score(all_preds, all_labels)
    print(f"Accuracy: {acc:.4f}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])
])

train = datasets.ImageFolder('./cats_and_dogs_filtered/train', transform=transform)
test = datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform=transform)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128)

model_with_custom_dropout = CNNWithCustomDropout(dropout_rate=0.5).to('cuda')
model_with_builtin_dropout = CNNWithBuiltInDropout(dropout_rate=0.5).to('cuda')

optimizer_with_custom_dropout = torch.optim.SGD(model_with_custom_dropout.parameters(), lr=0.001, momentum=0.9)
optimizer_with_builtin_dropout = torch.optim.SGD(model_with_builtin_dropout.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

print("Training model with custom dropout:")
train_model(model_with_custom_dropout, train_loader, optimizer_with_custom_dropout, criterion, epochs=10)
evaluate_model(model_with_custom_dropout, test_loader)

print("\nTraining model with built-in dropout:")
train_model(model_with_builtin_dropout, train_loader, optimizer_with_builtin_dropout, criterion, epochs=10)
evaluate_model(model_with_builtin_dropout, test_loader)

# Output :
# Training model with custom dropout:
# Epoch 1, Loss: 0.6874842718243599
# Epoch 2, Loss: 0.6758033409714699
# Epoch 3, Loss: 0.6595148593187332
# Epoch 4, Loss: 0.6426931619644165
# Epoch 5, Loss: 0.6283542402088642
# Epoch 6, Loss: 0.6106937490403652
# Epoch 7, Loss: 0.6043758392333984
# Epoch 8, Loss: 0.5856763049960136
# Epoch 9, Loss: 0.579024039208889
# Epoch 10, Loss: 0.5660783685743809
# Accuracy: 0.6390
#
# Training model with built-in dropout:
# Epoch 1, Loss: 0.6951404884457588
# Epoch 2, Loss: 0.6655388958752155
# Epoch 3, Loss: 0.6513420194387436
# Epoch 4, Loss: 0.635447233915329
# Epoch 5, Loss: 0.6160608120262623
# Epoch 6, Loss: 0.6029178835451603
# Epoch 7, Loss: 0.5867145247757435
# Epoch 8, Loss: 0.5830600373446941
# Epoch 9, Loss: 0.555221926420927
# Epoch 10, Loss: 0.5289735402911901
# Accuracy: 0.6780