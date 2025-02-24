import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

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

fashion_mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())
fasion_test_loader = DataLoader(fashion_mnist_testset, batch_size=50, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("../Week5/ModelFiles/mnist_model.pt")
model.to(device)

print("Model's state_dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t",model.state_dict()[param_tensor].size())
print()

model.eval()
correct = 0
total = 0
for i, vdata in enumerate(fasion_test_loader):
    tinputs, tlabels = vdata
    tinputs = tinputs.to(device)
    tlabels = tlabels.to(device)
    toutputs = model(tinputs)
    #Select the predicted class label which has the highest value in the output layer
    _, predicted = torch.max(toutputs, 1)
    print("True label:{}".format(tlabels))
    print('Predicted: {}'.format(predicted))
    total += tlabels.size(0)
    correct += (predicted == tlabels).sum()

accuracy = 100.0 * correct / total
print(f"The overall accuracy is {accuracy:.4f}")

# Output :
# The overall accuracy is 5.4100