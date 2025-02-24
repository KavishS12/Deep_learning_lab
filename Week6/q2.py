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

# Output :
# AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
#   (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=2, bias=True)
#   )
# )
# Epoch 0 - loss = 9.772921174764633
# Epoch 1 - loss = 5.895793199539185
# Epoch 2 - loss = 5.096952706575394
# Epoch 3 - loss = 4.604657739400864
# Epoch 4 - loss = 4.185598254203796
# Epoch 5 - loss = 4.17967164516449
# Epoch 6 - loss = 4.080375522375107
# Epoch 7 - loss = 3.7448328882455826
# Epoch 8 - loss = 3.6761072278022766
# Epoch 9 - loss = 3.7058169841766357
# Accuracy = 0.956
