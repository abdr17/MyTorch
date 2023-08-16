import mytorch as torch
import numpy as np
from mytorch import nn

x = np.ones((10, 4))
y = np.zeros((10, 4))
x = torch.MyArray(x, requires_grad=False)
y = torch.MyArray(y, requires_grad=True)

s = x + y
s.backward()
print(s.grad, y.grad)