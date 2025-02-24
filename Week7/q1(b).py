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

lambda_l2 = 0.01

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        l2_norm = 0
        for param in model.parameters():
            if param.requires_grad:
                l2_norm += torch.sum(param ** 2)
        loss += lambda_l2 * l2_norm
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
# Epoch 1, Loss: 0.4504995197057724
# Epoch 2, Loss: 0.2690520826727152
# Epoch 3, Loss: 0.22313756123185158
# Epoch 4, Loss: 0.20468539651483297
# Epoch 5, Loss: 0.19114950764924288
# Epoch 6, Loss: 0.1792193129658699
# Epoch 7, Loss: 0.16690047457814217
# Epoch 8, Loss: 0.16337243421003222
# Epoch 9, Loss: 0.1626289188861847
# Epoch 10, Loss: 0.1537919300608337
# Accuracy: 0.9540
