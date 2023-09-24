import torch

# 1. tensor initialisation
# 2. tensor maths
# 3. tensor indexing ✅
# 4. tensor reshaping

# basic indexing ->
batch_size = 16
features = 10
x = torch.rand((batch_size, features))
print(x)
print(x[0])  # first example
print(x[:, 0])  # first feature for all examples
print(x[2, 0:5])  # 3rd example's first 5 features
x[0, 0] = 20.34  # reassignment
print(x[0])


# fancy indexing ->
x = torch.arange(5, 15)  # [0, 1, ..., 14]
print(x)
indices = [1, 2, 3]
print(x[indices])  # prints the elements at index in indices list passed
xx = torch.rand((4, 3))
rows = torch.tensor([1, 0])
cols = torch.tensor([2, 0])
print(xx)
print(xx[rows, cols])  # returns xx[1, 2] and xx[0, 0]


# advanced indexing ->
y = torch.arange(5, 20)  # [5, 6, ..., 19]
print(y)
print(y[(y <= 8)])  # [5, 6, 7, 8]
print(y[(y < 8) | (y > 17)])  # [5, 6, 7, 18, 19]
print(y[y.remainder(2) == 1])  # returns those elements whose remainder with 2 is 1


# some useful operations ->
print(torch.where(y > 9, y, y*5))  # if element is greater than 9 then fine, else replace with y*5
print(torch.tensor([0, 0, 0, 2, 3, 4]).unique())  # [0, 2, 3, 4]
print(y.ndimension())  # returns the number of dimensions (somewhat y.shape ka shape)
print(y.numel())  # number of elements in y





