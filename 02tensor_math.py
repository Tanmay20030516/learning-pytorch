import torch

# 1. tensor initialisation
# 2. tensor maths ✅
# 3. tensor indexing
# 4. tensor reshaping

# addition
x = torch.tensor([3, 5, 8])
y = torch.tensor([7, 8, 2])
z1 = torch.empty(3)
# print(torch.add(x, y, out=z1))
# print(torch.add(x, y))
# print(x + y)

# subtraction
# print(x - y)

# division
# print(torch.true_divide(x, y)) # element wise division of x by y if equal shape

# inplace operations
t1 = torch.zeros(3)
t1.add_(x) # '_' means inplace addition
# print(t1)
t2 = torch.zeros(3)
t2 += x # is inplace
# print(t2)

# exponentiation (inplace)
t3 = x.pow(2)
# print(t3)
# print(x ** 2)
# print(x)

# comparisons
z = x > 0 # returns [True, True, True]
# print(z) # element wise comparisons

# matrix multiplications
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3_v1 = torch.mm(x1, x2)  # matrix multiplication of x1 and x2, out shape: 2x3
x3_v2 = x1.mm(x2)  # similar as line above
print(x3_v1, '\n', x3_v2)