import torch

tensor1 = torch.rand(7,7)

tensor2 = torch.rand(1,7)
tensor2_transposed = torch.transpose(tensor2,1,0)
multiplied = torch.matmul(tensor1,tensor2_transposed)
print(multiplied)

t1 = torch.rand(2,3).to('cuda')
t2 = torch.rand(2,3).to('cuda')
print(t1)
print(t2)
t2_transposed = torch.transpose(t2,1,0)
m2 = torch.matmul(t1,t2_transposed)
print(m2)

max_val = torch.max(t1)
min_val = torch.min(t1)
max_index = torch.argmax(t1)
min_index = torch.argmin(t1)

print("Max value and index : ", max_val,max_index)
print("Min value and index : ", min_val,min_index)
