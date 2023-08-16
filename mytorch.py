import numpy as np

class MyArray:
    def __init__(self, data, requires_grad=False, _children=()):
        
        # if isinstance(data, tuple):
        #     self.data = np.ndarray(data)
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            self.data = data
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self._backward = lambda : None
        if requires_grad == True:
            self.grad = MyArray(np.zeros(self.data.shape))
        self.shape = self.data.shape
    
    def __add__(self, other):
        if isinstance(other, MyArray):
            out = MyArray(self.data + other.data, _children=(self, other), requires_grad=(self.requires_grad or other.requires_grad))
            if(self.requires_grad == True or other.requires_grad == True):
                out.requires_grad = True
                def _backward():
                    if self.requires_grad == True:
                        self.grad += 1.0 * out.grad
                    if other.requires_grad == True:
                        other.grad += 1.0 * out.grad
                out._backward = _backward
        else:
            out = MyArray(self.data + other, requires_grad=(self.requires_grad))
            if(self.requires_grad == True):
                def _backward():
                    self.grad += 1.0 * out.grad
                out._backward = _backward
        
        return out

    def __mul__(self, other):

        # if other is a MyArray
        if isinstance(other, MyArray):
            if(self.data.shape[1] == other.data.shape[0]):
                out = MyArray(np.matmul(self.data, other.data), _children=(self, other), requires_grad=(self.requires_grad or other.requires_grad))
                if(self.requires_grad == True or other.requires_grad == True):
                    if(self.requires_grad == True and other.requires_grad == True):
                        def _backward():
                            temp = other.data @ out.grad.data.transpose()
                            self.grad.data += temp.transpose()
                            other.grad.data += self.data.transpose() @ out.grad.data
                        out._backward = _backward
                    else: 
                        def _backward():
                            if self.requires_grad==True:
                                self.grad.data += np.sum(other.data, axis=1)
                            else:
                                s = np.sum(self.data, axis=0)
                                s.shape = (s.shape[0], 1)
                                other.grad.data += s * out.grad.data
                        out._backward = _backward
                    # def _backward():
                    #     pass
                

        # if other is not a MyArray (i.e an int or a float)

        else:
            out = MyArray(self.data * other, requires_grad=(self.requires_grad))
            if(self.requires_grad == True):
                def _backward():
                    self.grad += float(other) * out.grad
                out._backward = _backward
        
        return out
    
    def __pow__(self, p):
        out = MyArray(self.data**p, _children=(self,), requires_grad=(self.requires_grad))
        if(self.requires_grad == True):
            def _backward():
                self.grad = (p * (self.data ** (p-1)) ) * out.grad.data
            out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        out = self + -1 * other
        return out
    
    def __neg__(self):
        return -1 * self

    def __rsub__(self, other): # other - self
        return other + (-self)

    def transpose(self):
        return MyArray(self.grad.data.transpose())

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = ones(self.shape)
        # self.grad = ones(self.grad.shape)
        for v in reversed(topo):
            v._backward()
    
    def __truediv__(self, other):
        return self * ((other) ** -1)
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1


    def log(self):
        return MyArray(np.log(self.data))
    
    def __repr__(self) -> str:
        return f' \nMyArray({self.data})' if (self.requires_grad == False) else f'MyArray({self.data}, requires_grad=true)'
    
def zeros(shape):
    return MyArray(np.zeros(shape=shape))

def ones(shape, requires_grad=False):
    return MyArray(np.ones(shape=shape), requires_grad=requires_grad)



def exp(tensor : MyArray):
    out = MyArray(np.exp(tensor.data))
    if(tensor.requires_grad == True):
        out.requires_grad = True
        def _backward():
            tensor.grad = tensor.data* out.grad

class nn:
    def __init__(self):
        pass

    def ReLU(tensor : MyArray):
        out = MyArray(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad)
        if(out.requires_grad):
            def _backward():
                tensor.grad = MyArray(out.grad.data * np.int_(np.not_equal(np.zeros(out.data.shape), out.data)))
            out._backward = _backward
        return out
    
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.w = MyArray(np.random.randn(out_features, in_features), requires_grad=True)
        self.b = MyArray(np.random.randn(1, out_features), requires_grad=True)
        self.w_trans = MyArray(self.w.data.transpose(),requires_grad=True)
        print(self.w_trans.shape)

    def __call__(self, x : MyArray):
        print(x.shape, self.w_trans.shape)
        out_1 = x * self.w_trans
        # out_2 = out_1 + self.b
        # self.w.grad.data = self.w_trans.grad.data.transpose()
        return out_1

    
class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, x : MyArray):
        out = MyArray(np.maximum(0, x.data), requires_grad=x.requires_grad)
        if(out.requires_grad):
            def _backward():
                x.grad = MyArray(out.grad.data * np.int_(np.not_equal(np.zeros(out.data.shape), out.data)))
            out._backward = _backward
        return out

