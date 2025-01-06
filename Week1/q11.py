import torch

torch.manual_seed(7)

tensor = torch.rand(1,1,1,10)
print(tensor.size())
print(tensor)

tensor2 = tensor.squeeze(dim=1)
print(tensor2.size())

tensor3 = tensor2.squeeze(dim=1)
print(tensor3.size())

tensor4 = tensor3.squeeze(dim=0)
print(tensor4.size())
print(tensor4)

