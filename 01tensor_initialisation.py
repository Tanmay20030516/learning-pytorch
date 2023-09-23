import torch

# 1. tensor initialisation ✅
# 2. tensor maths
# 3. tensor indexing
# 4. tensor reshaping




# declare a tensor
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# tensor2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda")
# print(tensor2)
device = "cuda" if torch.cuda.is_available() else "cpu"

ten3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
# print(ten3)
# print(ten3.dtype)
# print(ten3.device)
# print(ten3.shape)


# other initialisation methods
x = torch.empty(size=(3, 3)) # uninitialised / random data
y = torch.zeros((3, 3))
z = torch.rand((3, 3)) # from uniform distribution
q = torch.ones((3, 3)) # matrix of ones
w = torch.eye(4, 4) # identity matrix
e = torch.arange(start=0, end=11, step=2) # end is exclusive
r = torch.linspace(start=0.1, end=1, steps=10) # all numbers from 0.1 to 1 in 10 uniform steps
t = torch.empty(size=(2, 4)).normal_(mean=0, std=1) # initialise from a normal distribution
y1 = torch.empty(size=(3, 1)).uniform_(0, 2) # initialise form a uniform distribution
u = torch.diag(torch.ones(3)) # diagonal matrix
# print(x)
# print(y)
# print(z)
# print(q)
# print(w)
# print(e)
# print(r)
# print(t)
# print(y1)
# print(u)


# initialise and convert tensors to other types (int, double, float)
tensor = torch.arange(4) # tensor is [0, 4)
# print(tensor)
# print(tensor.bool()) # boolean
# print(tensor.short()) # int16
# print(tensor.long()) # int64
# print(tensor.half()) # float16
# print(tensor.float()) # float32
# print(tensor.double()) # float64


# array to tensor and tensor to array
import numpy as np
np_array = np.random.randn(5).reshape(1, -1)
to_tensor = torch.from_numpy(np_array)
np_array_back = to_tensor.numpy()
print(np_array)
print(to_tensor)
print(np_array_back)

