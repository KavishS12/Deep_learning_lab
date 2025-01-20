import torch

def f(x):
    return torch.exp(-x**2 - 2*x - torch.sin(x))

def analytical_gradient(x):
    fx = torch.exp(-x**2 - 2*x - torch.sin(x))
    return fx * (-2 * x - 2 - torch.cos(x))

x = torch.tensor(1.0, requires_grad=True)
fx = f(x)
fx.backward()
torch_gradient = x.grad
analytical_grad = analytical_gradient(x)

print(f"Function value at x: {fx.item()}")
print(f"PyTorch gradient: {torch_gradient.item()}")
print(f"Analytical gradient: {analytical_grad.item()}")
