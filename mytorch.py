# Implementation of autograd and linear, ReLU and sigmoid layers (very similar to pytorch)
from typing import Any
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

    def __getitem__(self, index):
        out = MyArray(np.array(self.data[index]), requires_grad=self.requires_grad)
        return out


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
            out = MyArray(self.data + other, requires_grad=(self.requires_grad), _children = (self,))
            if(self.requires_grad == True):
                def _backward():
                    self.grad += 1.0 * out.grad
                out._backward = _backward
        
        return out
    
    def element_wise_mul(self, other):
        out = MyArray(np.multiply(self.data, other.data), requires_grad=(self.requires_grad or other.requires_grad))
        if(self.requires_grad == True, other.requires_grad == True):
            def _backward():
                if (self.requires_grad == True) :
                    self.grad += MyArray(np.multiply(other.data, out.grad.data))
                if (other.requires_grad == True) :
                    other.grad += MyArray(np.multiply(self.data, out.grad.data))
        out._backward = _backward
        return out

    def __mul__(self, other, element_wise=False):
        # if other is a MyArray
        if isinstance(other, MyArray):
            if(self.data.shape[1] == other.data.shape[0]):
                out = MyArray(np.matmul(self.data, other.data), _children=(self, other), requires_grad=(self.requires_grad or other.requires_grad))
                if(self.requires_grad == True or other.requires_grad == True):
                    def _backward():
                        if self.requires_grad == True:
                            temp = other.data @ out.grad.data.transpose()
                            self.grad.data += temp.transpose()
                        if other.requires_grad ==True:
                            other.grad.data += self.data.transpose() @ out.grad.data
                    out._backward = _backward
        # if other is not a MyArray (i.e an int or a float)
        else:
            out = MyArray(self.data * other, requires_grad=(self.requires_grad), _children=(self,))
            if(self.requires_grad == True):
                def _backward():
                    self.grad += float(other) * out.grad
                out._backward = _backward
        return out
    
    def __pow__(self, p):
        out = MyArray(self.data**p, requires_grad=(self.requires_grad), _children = (self,))
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
    
    def square(self):
        out = MyArray(np.square(self.data), requires_grad=self.requires_grad, _children=(self,))
        if(self.requires_grad == True):
            def _backward():
                temp = (2 * self)
                self.grad = temp.element_wise_mul(out.grad)
            out._backward = _backward
        return out
    
    def sum(self, axis=0):
        out = MyArray(np.sum(self.data, axis=axis), requires_grad=self.requires_grad, _children=(self,))
        def _backward():
            try:
                self.grad += 1.0 * out.grad
            except:
                out.grad.data.shape = (out.grad.data.shape[0], 1)
                self.grad += 1.0 * out.grad
                out.grad.data.shape = (out.grad.data.shape[0], )
        out._backward = _backward
        return out
    
    def zero_grad(self):
        temp = MyArray(np.zeros(self.shape), requires_grad=True)
        self = temp

    def backward(self):

        # topologicalout.grad order all of the children in the graph
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

    def mean(self):
        out = MyArray([np.mean(self.data)], requires_grad=self.requires_grad, _children=(self,))
        def _backward():
            self.grad += 1/self.data.size * out.grad.data
        out._backward = _backward
        return out

    def log(self):
        return MyArray(np.log(self.data))
    
    def argmax(self, axis=1):
        out = MyArray(np.argmax(self.data, axis=axis))
    
    def __repr__(self) -> str:
        return f' \nMyArray({self.data})' if (self.requires_grad == False) else f'MyArray({self.data}, requires_grad=true)'
    
    
    
def zeros(shape):
    return MyArray(np.zeros(shape=shape))

def ones(shape, requires_grad=False):
    return MyArray(np.ones(shape=shape), requires_grad=requires_grad)


def exp(tensor : MyArray):
    out = MyArray(np.exp(tensor.data), requires_grad=tensor.requires_grad, _children=(tensor,))
    if(tensor.requires_grad == True):
        def _backward():
            tensor.grad = tensor.element_wise_mul(out.grad)
        out._backward = _backward
    return out

class nn:
    def __init__(self):
        pass

    def ReLU(tensor : MyArray):
        out = MyArray(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad, _children=(tensor,))
        if(out.requires_grad):
            def _backward():
                tensor.grad = MyArray(out.grad.data * np.int_(np.not_equal(np.zeros(out.data.shape), out.data)))
            out._backward = _backward
        return out
    
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.w = MyArray(np.random.randn(in_features, out_features), requires_grad=True)
        self.b = MyArray(np.random.randn(1, out_features), requires_grad=True)
        # self.w_trans = MyArray(self.w.data.transpose(),requires_grad=True)

    def __call__(self, x : MyArray):
        out = x * self.w
        out._prev = (x, self.w)    # Check if the code works without this line
        return out

    
class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, x : MyArray):
        out = MyArray(np.maximum(0, x.data), requires_grad=x.requires_grad, _children=(x,))
        if(out.requires_grad):
            def _backward():
                x.grad = MyArray(out.grad.data * np.int_(np.not_equal(np.zeros(out.data.shape), out.data)))
            out._backward = _backward
        return out
    
def sigmoid_function(arr):
    return 1 / (1 + np.exp(-1 * arr))
    

def Sigmoid(x):
    sig = MyArray(sigmoid_function(x.data), requires_grad=x.requires_grad, _children=(x,))
    if x.requires_grad == True:
        def _backward():
            x.grad += sig.element_wise_mul(1 - sig)
        sig._backward = _backward
    return sig

class SigmoidLayer:
    def __init__(self):
        global sig
    
    def __call__(self, x : MyArray):
        self.sig = MyArray(sigmoid_function(x), requires_grad=x.requires_grad, _children=(x,))
        if x.requires_grad == True:
            def _backward():
                x.grad += sig.element_wise_mul(1 - sig)
            self.sig._backward = _backward
        return self.sig
    
class MSELoss:
    def __init__(self):
        global out

    def __call__(self, arr_1 : MyArray, arr_2 : MyArray):
        out = ((arr_1-arr_2).square().mean())
        return out
    
class CrossEntropyLoss:
    def __init__(self):
        global out
    
    def __call__(self, pred : MyArray, des : MyArray):
        l = np.zeros((pred.shape[0], 1))
        for row in range(pred.shape[0]):
            l[row] = np.round(-1 * np.log( np.exp(pred.data[row][des.data[row]]) / np.sum(np.exp(pred.data[row])) ), 4)
        out = MyArray(l, requires_grad=pred.requires_grad, _children=(pred, des))
        def backward():
            grad = np.zeros(pred.shape)
            for row in range(pred.shape[0]):
                for col in range(pred.shape[1]):
                    if col == des[row]:
                        grad[row][col] = ((np.exp(pred.data[row][col])/np.sum(np.exp(pred.data[row][:]))))/pred.shape[0] + 1
                    else:
                        grad[row][col] = (np.exp(pred.data[row][col]) / np.sum(np.exp(pred.data[row][:])))/pred.shape[0]

            pred.grad = MyArray(grad)
        out._backward = backward
        return out.mean()
        # return (l)
    
def CrossEntropy(pred, des):
        l = np.zeros((pred.shape[0], 1))
        for row in range(pred.shape[0]):
            l[row] = np.round(-1 * np.log( np.exp(pred[row][des[row]]) / np.sum(np.exp(pred[row])) ), 4)
        # def backward():
        #     grad = np.zeros(pred.shape)
        #     for row in range(pred.shape[0]):
        #         for col in range(pred.shape[1]):
        #             if col == des[row]:
        #                 grad[row][col] = ((np.exp(pred[row][col])/np.sum(np.exp(pred[row][:]))) - 1)/pred.shape[0]
        #             else:
        #                 grad[row][col] = (np.exp(pred[row][col]) / np.sum(np.exp(pred[row][:])))/pred.shape[0]
        #     return np.round(grad, 4)
        # self._backward = backward
        # return (np.round(np.mean(l), 4))
        return (l)
            