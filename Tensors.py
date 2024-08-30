import numpy as np
import torch

print("as")
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
x_data = torch.Tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_np)
x_rand = torch.rand_like(x_ones, dtype=torch.float)

print(x_ones)
print(x_rand.device)

if torch.cuda.is_available():
    