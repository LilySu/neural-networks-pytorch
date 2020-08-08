import torch

# a tensor can have different dimensions
# create an empty tensor
# x = torch.empty(1)
# result: tensor([7.5704e-27])

# x = torch.empty(3)
# result: tensor([1.2121e-42, 0.0000e+00, 3.0178e-38])

# x = torch.empty(2, 3)
# result: tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# x = torch.empty(2, 2, 3)
# result:tensor([[[0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [0.0000e+00, 2.8026e-45, 0.0000e+00]],
#
#         [[1.1210e-44, 0.0000e+00, 1.4013e-45],
#          [0.0000e+00, 0.0000e+00, 0.0000e+00]]])

# torch.ones
# torch.zeros
# torch.ones(2,2, dtype=type.double)

# print(torch.tensor([2.5, 0.1]))

# x = torch.rand(2,2)
# y = torch.rand(2,2)
# print(x)
# print(y)
# z = x + y
# z = torch.add(x,y)
# print(z)
#
# y.add_(x) # adds all elements of x to y
# # trailing underscore is an inplace transform
# z = torch.mul(x,y)

# gradient calculation
# x = torch.randn(3, requires_grad=True)
# # print(x)
# y = x + 2
# # print(y)
# z = y * y * 2
# z = z.mean()
# z.backward() # dz/dx
# print(x.grad)

# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z.backward(v)

# back propagation
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)
y_hat = w * x
loss = (y_hat - y)**2
# print(loss)
loss.backward() # whole gradient computation
# print(w.grad)



if __name__ == '__main__':
    # print(x)
    # print(z)
    # print(x.grad)
    # print(z)
    print(loss)
    print(w.grad)