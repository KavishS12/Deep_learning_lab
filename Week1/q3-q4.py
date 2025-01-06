import torch
import numpy as np
from sympy import shape

tensor = torch.rand((2,3)).to('cuda')
print(tensor)

print(tensor[1])
print(tensor[1][0])

numpy_array = tensor.cpu().numpy()
print(type(numpy_array))
print(numpy_array)

a = np.array([[1,2,3],[4,5,6]])
print(a)
tensor2 = torch.from_numpy(a)
print(type(tensor2))
print(tensor2)
