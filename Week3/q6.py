import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[3.,8.],[4.,5.],[5.,7.],[6.,3.],[2.,1.]])
y = torch.tensor([-3.7,3.5,2.5,11.5,5.7]).view(-1, 1)

model = nn.Linear(in_features=2, out_features=1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch : {epoch}, Loss: {loss.item()}")

print()
print(f"Final weight for X1: {model.weight[0][0].item()}, Final weight for X2: {model.weight[0][1].item()}")
print(f"Final bias: {model.bias.item()}")

x_test = torch.tensor([[3, 2]], dtype=torch.float32)
y_test_pred = model(x_test)
print()
print(f"Predicted Y for X1=3 and X2=2: {y_test_pred.item()}")