import torch
import torch.nn as nn


data = torch.randn(100,3,32,32)


conv = nn.Conv2d(in_channels = 3, out_channels = 53, kernel_size=5, stride=1, padding=1)
output = conv(data)
print(output)
print(output.shape)