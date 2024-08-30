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
print(x_data.device)

if torch.cuda.is_available():
    x_data = x_data.to(device='cuda')
    print(x_data.device)

tensor = torch.ones(4, 4)
tensor[:, -1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor], dim=1)
print(t1)

y1 = t1 @ t1.T
y2 = t1.matmul(t1.T)

y3 = torch.rand_like(y1)
torch.matmul(t1, t1.T, out=y3)
print(y1, "\n", y2, "\n", y3)
