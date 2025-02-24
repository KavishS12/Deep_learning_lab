import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # Resize to 224x224 (AlexNet's required size)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# note
train = datasets.ImageFolder(
    './cats_and_dogs_filtered/train', transform=transform['train'])

test = datasets.ImageFolder(
    './cats_and_dogs_filtered/validation', transform=transform['val'])
train_loader = DataLoader(train, batch_size=128, shuffle=True)
test_loader = DataLoader(test, batch_size=128, shuffle=True)

# note
model = models.alexnet(weights='IMAGENET1K_V1')

# note very important to freeze
for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, 2)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

model.to('cuda')

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch} - loss = {running_loss}')

all_preds, all_target = [], []
model.eval()
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to('cuda'), target.to('cuda')
        output = model(input)
        val, index = torch.max(output, dim=1)
        all_preds.extend(index.to('cpu'))
        all_target.extend(target.to('cpu'))

acc = accuracy_score(all_preds, all_target)
print(f"Accuracy = {acc}")

