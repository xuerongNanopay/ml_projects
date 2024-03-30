# Initializing Tensor

import torch

DEVICE = "cude" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]],
    dtype=torch.float32,
    device=DEVICE,
    requires_grad=True,
)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(torch.cuda.is_available())
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size=(3, 3))
print(x)
x = torch.zeros((3, 3))
print(x)
# Uniform distribution from 0 to 1
x = torch.rand((3, 3))
print(x)
x = torch.ones((3, 3))
print(x)
x = torch.eye(5,5)
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
print(x)
x = torch.diag(torch.ones(3))
print(x)

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) # boolean True/False
print(tensor.short()) # int16
print(tensor.long()) # int64 (Important)
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double()) # float64

# Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

# Tensor Match & Comparison
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
z = x + y

# Substraction
z = x - y

# Division
z = torch.true_divide(x, y)
print(z)
z = torch.true_divide(x, 2)
print(z)

# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x # but not t = t + x
print(t)

# Exponentiation
z = x.pow(2)
z = x ** 2
print(z)

# Matrix comparison
z = z > 0
print(z)
z = z < 0
print(z)

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
print(x3)
x4 = x1.mm(x2)
print(x4)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)

# element wize mult.
z = x * y
print(z)

# dot product
z = torch.dot(x, y)
print(z, z.data, z.shape)

# batch matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm.shape)

# Example of Broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operations
x = torch.tensor([[1,1,1],[0,0,0],[2,2,2]])
sum_x = torch.sum(x, dim=0)
print(sum_x)

sum_x = torch.sum(x, dim=1)
print(sum_x)

values, indices = torch.max(x, dim=0)
print(values, indices)
values, indices = torch.min(x, dim=0)
print(values, indices)
abs_x = torch.abs(x)
print(abs_x)

z = torch.argmax(x, dim=0)
print(z)
z = torch.argmin(x, dim=0)
print(z)
z = torch.argmax(x, dim=1)
print(z)
z = torch.argmin(x, dim=1)
print(z)

x = torch.tensor([[1,1,1],[0,1,0],[2,2,2]])
print(x.shape)
mean_x = torch.mean(x.float(), dim=0)
print(mean_x)
mean_x = torch.mean(x.float(), dim=1)
print(mean_x)

x = torch.tensor([[1,1,1],[0,1,0],[2,2,2]])
y = torch.tensor([[1,1,1],[0,1,0],[2,2,2]])
z = torch.eq(x, y)
print(z)

sorted_y, indices = torch.sort(y, dim=0, descending=False)
print(sorted_y)
print(indices)

z = torch.clamp(x, min=0, max=1)
print(z)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
print(z)
z = torch.all(x)
print(z)

if __name__ == "__main__":
    pass