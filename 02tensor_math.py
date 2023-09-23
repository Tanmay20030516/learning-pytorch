import torch

# 1. tensor initialisation
# 2. tensor maths ✅
# 3. tensor indexing
# 4. tensor reshaping

# addition ->
x = torch.tensor([3, 5, 8])
y = torch.tensor([7, 8, 2])
z1 = torch.empty(3)
# print(torch.add(x, y, out=z1))
# print(torch.add(x, y))
# print(x + y)

# subtraction ->
# print(x - y)

# division ->
# print(torch.true_divide(x, y))  # element wise division of x by y if equal shape

# inplace operations ->
t1 = torch.zeros(3)
t1.add_(x)  # '_' means inplace addition
# print(t1)
t2 = torch.zeros(3)
t2 += x  # is inplace
# print(t2)

# exponentiation (inplace) ->
t3 = x.pow(2)
# print(t3)
# print(x ** 2)
# print(x)

# comparisons ->
z = x > 0  # returns [True, True, True]
# print(z)  # element wise comparisons

# matrix multiplications ->
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3_v1 = torch.mm(x1, x2)   # matrix multiplication of x1 and x2, out shape: 2x3
x3_v2 = x1.mm(x2)   # similar as line above
# print(x3_v1, '\n', x3_v2)

# matrix exponentiation ->
matrix_exp = torch.rand(4, 4)
# print(matrix_exp, '\n', matrix_exp.matrix_power(3))  # A, A^3

# matrix elementwise multiplication ->
# print(x * y)  # returns [3*7, 5*8, 8*2]

# matrix(vector) dot product ->
# print(torch.dot(x, y))  # returns 3*7 + 5*8 + 8*2

# batch matrix multiplication ->
batch_size = 8
m = 3
n = 2
o = 3
ten1 = torch.rand((batch_size, m, n))
ten2 = torch.rand((batch_size, n, o))
out_bmm = torch.bmm(ten1, ten2)  # of shape (batch_size, m, o)
# print(ten1, '\n', ten2, '\n', out_bmm)

# broadcasting ->
x1 = torch.rand((5, 5))
x2 = torch.rand((5, 1))
# print(x1-x2)  # (5, 5) shape, column of x2 was expanded (repeated) to match column number of x1
# print(x1 ** x2)  # element wise exponentiation for every column

# other useful mathematical operations ->
# x = torch.tensor([3, 5, 8])
sum_x = torch.sum(x, dim=0)  # Sum of x across dim=0 (which is the only dim in our case)
# print(sum_x)  # outputs tensor(16)
values1, indices1 = torch.max(x, dim=0)  # returns the max values and their indices; Can also do x.max(dim=0)
# print(values1, ' ', indices1) # returns 8 2
values2, indices2 = torch.min(x, dim=0)  # returns the min values and their indices; Can also do x.min(dim=0)
# print(values2, ' ', indices2) # returns 3 0
abs_x = torch.abs(x)  # returns x where abs function has been applied to every element
# print(abs_x)
z_imax = torch.argmax(x, dim=0)  # Gets index of the maximum value
# print(z_imax)
z_imin = torch.argmin(x, dim=0)  # Gets index of the minimum value
# print(z_imin)
mean_x = torch.mean(x.float(), dim=0)  # mean requires x to be float
# print(mean_x)
zt = torch.eq(x, y)  # Element wise comparison for equality, in this case z = [False, False, False]
# print(zt)
sorted_y, indices = torch.sort(y, dim=0, descending=False)
#      0  1  2
# y = [7, 8, 2]
# print(sorted_y, '\n', indices) # indices = [2 0 1]
zz = torch.clamp(x, min=0)  # like gradient clipping by value
# All values < 0 set to 0 and values > 0 unchanged (this is exactly ReLU function)
# If you want to values over max_val to be clamped, do torch.clamp(x, min=min_val, max=max_val)
# this would make the values lie between min_val and max_val

xx = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)  # True/False values
za = torch.any(xx)  # will return True, can also do x.any() instead of torch.any(x)
zb = torch.all(xx)  # will return False (since not all are True), can also do x.all() instead of torch.all()
# print(za, '\n', zb)







