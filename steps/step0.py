import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        # 如果不是 ndarray 的数据，报错
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError('{} is not a numpy array'.format(data))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, creator):
        self.creator = creator

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        functions = [self.creator]
        while functions:
            f = functions.pop()
            x, y = f.inputs, f.outputs
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                functions.append(x.creator)


class Function:
    def __call__(self, inputs):
        x = inputs.data
        y = self.forward(x)
        outputs = Variable(as_array(y))  # 避免 0 维的 array的运算问题
        outputs.set_creator(self)
        self.inputs = inputs  # 正向传播保存值
        self.outputs = outputs
        return outputs

    # 正向传播
    def forward(self, inputs):
        raise NotImplementedError

    # 反向传播
    def backward(self, grad):
        raise NotImplementedError


class Square(Function):
    def forward(self, inputs):
        return inputs ** 2

    def backward(self, grad):
        x = self.inputs.data
        return 2 * x * grad   # 对当前变量的导数 * 后面传过来的导数值


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, inputs):
        return np.exp(inputs)

    def backward(self, grad):
        x = self.inputs.data
        return np.exp(x) * grad


def exp(x):
    return Exp()(x)


# 中心差分函数,对函数 f 在 x 处进行求导
def numeric_diff(f, x, esp=1e-4):
    x0 = Variable(x.data + esp)
    x1 = Variable(x.data - esp)
    return (f(x0).data - f(x1).data) / (2.0 * esp)


def function1(inputs):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(inputs)))


if __name__ == '__main__':
    v = Variable(np.array([2]))
    A = Square()
    B = Exp()
    C = Square()
    a = A(v)
    b = B(a)
    y = C(b)
    y.grad = np.array(1.0)
    y.backward()
    print(v.grad)
