import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])])
train = datasets.ImageFolder('./cats_and_dogs_filtered/train', transform= transforms)
test = datasets.ImageFolder('./cats_and_dogs_filtered/validation', transform= transforms)
train_loader = DataLoader(train, batch_size= 64, shuffle= True)
test_loader = DataLoader(test, batch_size= 64)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), stride= (1, 1), padding=(1, 1)),
                                 nn.Conv2d(32, 64, kernel_size= (3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                 nn.MaxPool2d(kernel_size= 2, stride= 2, padding= 0),
                                 nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.MaxPool2d(kernel_size=2, stride=2))
        self.classify_head = nn.Sequential(nn.Flatten(),
                                           nn.Linear(128 * 28 * 28, 512, bias= True),
                                           nn.Dropout(0.5),
                                           nn.Linear(512, 2, bias= True))
    def forward(self, x):
        return self.classify_head(self.net(x))

model = CNN()
model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

patience = 5
curr_patience = 0
best_val_loss = float('inf')
for epoch in range(10):
    model.train()
    for input, target in train_loader:
        input, target = input.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to('cuda'), target.to('cuda')
            output = model(input)
            loss = criterion(output, target)
            val_loss += loss.item()
    val_loss /= len(test_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        curr_patience = 0
    elif curr_patience == patience:
        break
    else:
        curr_patience += 1

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for input, target in test_loader:
        input, target = input.to('cuda'), target.to('cuda')
        output = model(input)
        val, ind = torch.max(output, dim=1)
        all_preds.extend(ind.to('cpu'))
        all_labels.extend(target.to('cpu'))
from sklearn.metrics import accuracy_score

print(accuracy_score(all_preds, all_labels))


