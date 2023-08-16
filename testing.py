import mytorch as torch
import numpy as np
from mytorch import nn

x = np.ones((5, 3))
w = np.ones((3, 3))
x = torch.MyArray(x, requires_grad=True)
w = torch.MyArray(w, requires_grad=True)

s = x * w
s.backward()
print(s.grad, w.grad, x.grad)