import torch
from numpy.array_api import squeeze

print(torch.cuda.is_available())

tensor = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(tensor.shape)
print(tensor)

reshaped = torch.reshape(tensor,(1,9))
print(reshaped.shape)
print('Reshaped : ',reshaped)

tensor1 = torch.tensor([1,2,3])
tensor2 = torch.tensor([10,11,12])
stacked = torch.stack([tensor1,tensor2],dim=1)
print(stacked)

unsqueezed = tensor1.unsqueeze(dim=1)
print(unsqueezed.shape)
print(unsqueezed)
squeezed = unsqueezed.squeeze(dim=1)
print(squeezed.shape)
print(squeezed)
