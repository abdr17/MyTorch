import torch
# a = [[1, 2, 3, 4], [2, 3, 4, 1]]
# b = [[1, 2, 3], [2, 3, -1], [1, 2, 9], [1, 4, -3]]
# u = [4, 1, 1]
# v = [1, 5, 6]

# a = torch.Tensor(a)
# a.requires_grad=True
# b = torch.Tensor(b)
# b.requires_grad=True
# u = torch.Tensor(u)
# u.requires_grad=True
# v = torch.Tensor(v)
# v.requires_grad=True

# c = a @ b
# s = torch.sum(a)
# s.backward()
# print(a.grad)
x = torch.ones((15, 3), requires_grad=True)
w = torch.ones((3, 3), requires_grad=True)
# b = torch.ones((10,4), requires_grad=True) 
out_1 = (x @ w)
# l1 = torch.nn.Linear(in_features=784, out_features=10)
# out = l1(x)
# print(out_2)
out_2 = torch.sum(out_1)
out_2.backward()
print(w.grad)
# print(l1.bias.grad)