import torch

x = torch.tensor(1.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)
w = torch.tensor(2.,requires_grad=True)

u = w*x
v = u+b
a = torch.sigmoid(v)

a.backward()
print(w.grad)