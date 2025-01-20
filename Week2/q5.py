import torch

x = torch.tensor(2.0, requires_grad=True)
y = 8 * x**4 + 3 * x**3 + 7 * x**2 + 6 * x + 3
y.backward()
torch_gradient = x.grad 

def analytical_gradient(x):
    return 32 * x**3 + 9 * x**2 + 14 * x + 6

analytical_grad = analytical_gradient(1.0)

print(f"PyTorch gradient: {torch_gradient.item()}")
print(f"Analytical gradient: {analytical_grad}")
