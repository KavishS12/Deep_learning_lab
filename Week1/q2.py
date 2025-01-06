# torch.permute()
import torch

tensor = torch.rand((2,3,4))
print(tensor)

# change dimension to (3,2,4)
permuted1 = torch.permute(tensor,(1,0,2))
print(permuted1)

# change dimension to (4,3,2)
permuted2 = torch.permute(tensor,(2,1,0))
print(permuted2)