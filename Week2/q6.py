import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

a = 2 * x
b = torch.sin(y)
c = a / b
d = z * c
e = torch.log(1 + d)
f = torch.tanh(e)

f.backward()

print(f"Gradient of f with respect to x: {x.grad.item()}")
print(f"Gradient of f with respect to y: {y.grad.item()}")
print(f"Gradient of f with respect to z: {z.grad.item()}")
