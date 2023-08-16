import torch
# a = [[2, -1, 3], [-1, 3, 2], [2, 2, 2], [4, 3, -4], [-5, -1, 8]]
# b = [[2, 1, 1], [3, 2, 1],[1, 2, 3]]
# m = [[3, 4], [4, 5],[5, 6]]
# # u = [4, 1, 1]
# v = [1, 5, 6]

x = torch.ones((10, 10), requires_grad=True)
w = torch.ones((10, 10), requires_grad=True)
m = torch.ones((10, 10), requires_grad=True)
m = m * 2

# a = torch.Tensor(a)
# a.requires_grad=True
# b = torch.Tensor(b)
# b.requires_grad=True
# m = torch.Tensor(m)
# u.requires_grad=True
# v = torch.Tensor(v)
# v.requires_grad=True

# c = a @ b
# s = torch.sum(a)
# s.backward()
# print(a.grad)
# x = torch.ones((15, 3), requires_grad=True)
# w = torch.ones((3, 3), requires_grad=True)
# b = torch.ones((10,4), requires_grad=True) 
out_1 = (x+ w) @ m
out_2 = torch.sum(out_1)
# l1 = torch.nn.Linear(in_features=784, out_features=10)
# out = l1(x)
# print(out_2)
# out_3 = torch.sum(out_2)
out_2.backward()
print(x.grad, w.grad)
# print(l1.bias.grad)