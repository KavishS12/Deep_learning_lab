import torch
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self):
        self.w = torch.tensor(1.0, requires_grad=True)
        self.b = torch.tensor(1.0, requires_grad=True)

    def forward(self,xj):
        return self.w * xj + self.b

    def update(self,learning_rate):
        with torch.no_grad():
            self.w -= learning_rate * self.w.grad
            self.b -= learning_rate * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

def criterion(yj,yp):
        return (yj - yp) ** 2

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

learning_rate = torch.tensor(0.001)

model = RegressionModel()
loss_list = []

for epoch in range(100):
    loss = 0.0
    for i in range(len(x)) :
        y_p = model.forward(x[i])
        loss += criterion(y[i],y_p)
    loss = loss/len(x)
    loss_list.append(loss.item())
    print(f"Epoch : {epoch} , loss = {loss}")

    loss.backward()

    model.update(learning_rate)
    model.reset_grad()

print(f'Final weight (w): {model.w.item()}')
print(f'Final bias (b): {model.b.item()}')

plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.grid()
plt.show()