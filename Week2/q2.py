import torch

x = torch.tensor(7.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)
w = torch.tensor(4.,requires_grad=True)

u = w*x
v = u+b
a = torch.relu(v)

a.backward()
print(w.grad)