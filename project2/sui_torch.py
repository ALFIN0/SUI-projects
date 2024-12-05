import numpy as np

class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = str(self.back_op.__class__.__name__)
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'
    
    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        if deltas is not None:
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {self.value.shape}, got {deltas.shape}'
            self.grad = deltas
            if self.back_op:
                self.back_op.backward(deltas)
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            self.grad = np.ones_like(self.value)
            if self.back_op:
                self.back_op.backward(self.grad)

class SumOp:
    def __init__(self, tensor):
        self.tensor = tensor

    def backward(self, grad):
        self.tensor.grad += np.ones_like(self.tensor.value) * grad

def sui_sum(tensor):
    """Sums all elements in a tensor."""
    if not isinstance(tensor, Tensor):
        raise TypeError("Inputs to sui_sum must be Tensor objects")
    
    value = np.sum(tensor.value)
    op = SumOp(tensor)
    return Tensor(value, op)

class AddOp:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad):
        self.a.grad += grad
        self.b.grad += grad

def add(a, b):
    """Element-wise addition."""   
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("Inputs to add must be Tensor objects")

    value = a.value + b.value
    op = AddOp(a, b)
    return Tensor(value, op)

class SubtractOp:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad):
        self.a.grad += grad
        self.b.grad -= grad

def subtract(a, b):
    """Element-wise subtraction."""
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("Inputs to subtract must be Tensor objects")
    
    value = a.value - b.value
    op = SubtractOp(a, b)
    return Tensor(value, op)

class MultiplyOp:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad):
        self.a.grad += grad * self.b.value
        self.b.grad += grad * self.a.value

def multiply(a, b):
    """Element-wise multiplication."""
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("Inputs to multiply must be Tensor objects")
    
    value = a.value * b.value
    op = MultiplyOp(a, b)
    return Tensor(value, op)

class ReLUOp:
    def __init__(self, tensor):
        self.tensor = tensor

    def backward(self, grad):
        self.tensor.grad += grad * (self.tensor.value > 0)

def relu(tensor):
    """ReLU activation function."""
    if not isinstance(tensor, Tensor):
        raise TypeError("Input to relu must be a Tensor object")
    
    value = np.maximum(0, tensor.value)
    op = ReLUOp(tensor)
    return Tensor(value, op)

class DotProductOp:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad):
        self.a.grad += np.dot(grad, self.b.value.T)
        self.b.grad += np.dot(self.a.value.T, grad)

def dot_product(a, b):
    """Matrix multiplication."""
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("Inputs to dot_product must be Tensor objects")
    
    value = np.dot(a.value, b.value)
    op = DotProductOp(a, b)
    return Tensor(value, op)
