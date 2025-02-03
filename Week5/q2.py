import torch
import torch.nn as nn

image = torch.rand(6,6)
print("image=", image)

image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)

print("image=", image)
kernel = torch.ones(3,3)
print("kernel=", kernel)

kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)

out_image = nn.Conv2d(6,3,(3,3))
print("outimage=", out_image)
