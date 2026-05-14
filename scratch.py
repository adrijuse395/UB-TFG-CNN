import torch
import torch.nn as nn
from src.decompositions.svd import SVDDecomposedLayer

torch.manual_seed(42)

# Create a mock Conv2d layer
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)

# Compress it using SVD
svd_layer = SVDDecomposedLayer.from_layer(conv, rank=32)

# Test forward pass
x = torch.randn(1, 64, 32, 32)
y_orig = conv(x)
y_comp = svd_layer(x)

print("Original output shape:", y_orig.shape)
print("Compressed output shape:", y_comp.shape)
print("Difference norm:", torch.norm(y_orig - y_comp).item())

# Test Linear layer
lin = nn.Linear(512, 128)
svd_lin = SVDDecomposedLayer.from_layer(lin, rank=16)
x_lin = torch.randn(1, 512)
y_orig_lin = lin(x_lin)
y_comp_lin = svd_lin(x_lin)

print("Original linear shape:", y_orig_lin.shape)
print("Compressed linear shape:", y_comp_lin.shape)
print("Linear difference norm:", torch.norm(y_orig_lin - y_comp_lin).item())
