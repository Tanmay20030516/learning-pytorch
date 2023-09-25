import torch

# 1. tensor initialisation
# 2. tensor maths
# 3. tensor indexing
# 4. tensor reshaping ✅

# basic reshaping ->
x = torch.arange(8)
x_4x2_v = x.view(4, 2)  # .view() works when reshaping array is contiguously allocated
x_4x2_r = x.reshape(4, 2)  # .reshape() creates a copy to make it contiguously store the tensor, and then reshape it
print(x)
print(x_4x2_v)
print(x_4x2_r)
y = x_4x2_v.t()  # transpose
print(y.is_contiguous())  # False
try:
    print(y.view(8))  # gives error
except Exception as exc:
    (print(exc))  # view size is not compatible with input tensor's size and stride
    # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
print(y.contiguous().view(8))  # error-free
print(y.reshape(8))  # no need to use .contiguous() now


# other types of reshaping ->
x1 = torch.rand(3, 6)
x2 = torch.rand(3, 6)
print(torch.cat((x1, x2), dim=0).shape)  # shape: 6x6; dim ~ axis argument
print(torch.cat((x1, x2), dim=1).shape)  # shape: 3x12

z = x1.view(-1)  # unrolls into flat tensor
print(z, '\n', z.shape)


# batch reshaping ->
m = 16
x = torch.rand((m, 3, 6))
z = x.view(m, -1)  # keeps the 1st dim untouched, and unrolls other two dims into a flat tensor
print(z.shape)

z = x.permute(0, 2, 1)  # switch between 3rd and 2nd dimension
print(z.shape)  # shape = (16, 6, 3)

z = torch.chunk(x, chunks=2, dim=1)  # splits the dim into 2 chunks,
# if unequal, then accordingly into ceil(dim_val/2) and floor(dim_val/2) sized
print(len(z))
print(z[0].shape)
print(z[1].shape)


# adding a new dimension ->
x = torch.arange(10)  # shape is [10]
# add a new dim so we have 1x10
print(x.unsqueeze(0).shape)  # 1x10
print(x.unsqueeze(1).shape)  # 10x1

# Let's say we have x which is 1x1x10, and we want to remove a dim, so we get 1x10
x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
print(x.shape)
z1 = x.squeeze(1)  # shape = 1x10
z2 = x.squeeze(0)  # shape = 1x10
print(z1.shape)
print(z2.shape)
