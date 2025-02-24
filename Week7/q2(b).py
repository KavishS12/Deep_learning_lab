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

model = models.alexnet(weights='IMAGENET1K_V1')

# Freeze all the layers so only the final classifier is trained
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer for binary classification (2 classes)
model.classifier[6] = nn.Linear(4096, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

lambda_l1 = 0.01

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        l1_norm = 0
        for param in model.parameters():
            if param.requires_grad:
                l1_norm += torch.sum(torch.abs(param))
        loss += lambda_l1 * l1_norm
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)
        output = model(input)
        _, predicted = torch.max(output, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

acc = accuracy_score(all_preds, all_labels)
print(f"Accuracy: {acc:.4f}")

# Output
# Epoch 1, Loss: 1.0875614546239376
# Epoch 2, Loss: 0.8893568217754364
# Epoch 3, Loss: 0.8441766984760761
# Epoch 4, Loss: 0.8010476343333721
# Epoch 5, Loss: 0.7745503671467304
# Epoch 6, Loss: 0.755395881831646
# Epoch 7, Loss: 0.731435913592577
# Epoch 8, Loss: 0.7162365466356277
# Epoch 9, Loss: 0.6990856416523457
# Epoch 10, Loss: 0.6841495558619499
# Accuracy: 0.9610

