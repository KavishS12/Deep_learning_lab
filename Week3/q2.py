import torch

x = torch.tensor([2.,4.])
y = torch.tensor([20.,40.])

w = torch.tensor(1.,requires_grad=True)
b = torch.tensor(1.,requires_grad=True)

alpha = torch.tensor(0.001)

for epoch in range(2) :
    loss= 0.0
    for i in range(len(x)) :
        a = w * x[i]
        y_p = a + b
        loss += (y_p - y[i]) ** 2
    loss = loss/len(x)

    loss.backward()
    print(f"Epoch : {epoch}")
    print(f"w_grad = {w.grad} , b_grad = {b.grad} , loss = {loss}")
    with torch.no_grad() :
        w -= alpha * w.grad
        b -= alpha * b.grad
    print(f"w = {w} , b = {b}")
    print()

    w.grad.zero_()
    b.grad.zero_()