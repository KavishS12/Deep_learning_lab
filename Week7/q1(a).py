import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train = datasets.ImageFolder('./cats_and_dogs_filtered/train', transform=transform)
test = datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform=transform)

train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128)

model = models.alexnet(weights='IMAGENET1K_V1')

# Freeze convolutional layers, but keep the fully connected layers trainable
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# Modify the classifier for binary classification
model.classifier[6] = nn.Linear(4096, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

all_preds, all_labels = [], []
model.eval()

with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)
        output = model(input)
        val, ind = torch.max(output, dim=1)
        all_preds.extend(ind.to('cpu').numpy())
        all_labels.extend(target.to('cpu').numpy())

acc = accuracy_score(all_preds, all_labels)
print(f'Accuracy: {acc:.4f}')

#Output
# Epoch 1, Loss: 0.4116
# Epoch 2, Loss: 0.2365
# Epoch 3, Loss: 0.1992
# Epoch 4, Loss: 0.1767
# Epoch 5, Loss: 0.1653
# Epoch 6, Loss: 0.1575
# Epoch 7, Loss: 0.1492
# Epoch 8, Loss: 0.1434
# Epoch 9, Loss: 0.1410
# Epoch 10, Loss: 0.1284
# Accuracy: 0.9570

