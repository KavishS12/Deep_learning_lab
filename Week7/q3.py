import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])
])

train = datasets.ImageFolder('./cats_and_dogs_filtered/train', transform=transform)
test = datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform=transform)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128)

class CNNWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer with specified rate

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

class CNNWithoutDropout(nn.Module):
    def __init__(self):
        super(CNNWithoutDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
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

model_with_dropout = CNNWithDropout(dropout_rate=0.5).to('cuda')
model_without_dropout = CNNWithoutDropout().to('cuda')

optimizer_with_dropout = torch.optim.SGD(model_with_dropout.parameters(), lr=0.001, momentum=0.9)
optimizer_without_dropout = torch.optim.SGD(model_without_dropout.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

print("Training model with dropout:")
train_model(model_with_dropout, train_loader, optimizer_with_dropout, criterion, epochs=10)
evaluate_model(model_with_dropout, test_loader)

print("\nTraining model without dropout:")
train_model(model_without_dropout, train_loader, optimizer_without_dropout, criterion, epochs=10)
evaluate_model(model_without_dropout, test_loader)

#Output :
# Training model with dropout:
# Epoch 1, Loss: 0.6879289969801903
# Epoch 2, Loss: 0.6745897606015205
# Epoch 3, Loss: 0.6543623507022858
# Epoch 4, Loss: 0.6321450807154179
# Epoch 5, Loss: 0.626698974519968
# Epoch 6, Loss: 0.604354489594698
# Epoch 7, Loss: 0.582191813737154
# Epoch 8, Loss: 0.58086758852005
# Epoch 9, Loss: 0.5893617495894432
# Epoch 10, Loss: 0.5602136105298996
# Accuracy: 0.6510
#
# Training model without dropout:
# Epoch 1, Loss: 0.6916178055107594
# Epoch 2, Loss: 0.6711471676826477
# Epoch 3, Loss: 0.6432548761367798
# Epoch 4, Loss: 0.6286686174571514
# Epoch 5, Loss: 0.601884551346302
# Epoch 6, Loss: 0.5830974765121937
# Epoch 7, Loss: 0.5728394314646721
# Epoch 8, Loss: 0.5461182724684477
# Epoch 9, Loss: 0.527393463999033
# Epoch 10, Loss: 0.510452413931489
# Accuracy: 0.6580
