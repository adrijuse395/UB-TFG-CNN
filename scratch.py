import torch
import tensorly as tl
from tensorly.decomposition import parafac
tl.set_backend('pytorch')

W = torch.randn(64, 3, 3, 3)
rank = 16
cp_weights, factors = parafac(W, rank=rank, init='svd')

print(f"CP Weights: {cp_weights.shape}")
print(f"Factors type: {type(factors)}")
for i, f in enumerate(factors):
    print(f"Factor {i} shape: {f.shape}")
