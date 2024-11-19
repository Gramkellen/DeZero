from globaldefine import Variable, Function
import numpy as np


class Add(Function):
    def forward(self, a, b):       # 成功与否有待验证
        return a+b

    def backward(self, grad):
        return grad, grad


def add(a, b):
    return Add()(a, b)


class Square(Function):
    def forward(self, inputs):
        return inputs ** 2

    def backward(self, grad):
        x = self.inputs[0].data
        return 2 * x * grad   # 对当前变量的导数 * 后面传过来的导数值


def square(x):
    return Square()(x)


if __name__ == '__main__':
    a = Variable(np.array(10))
    b = Variable(np.array(20))
    y = add(square(a), square(b))
    y.backward()
    print(y.data)
    print(a.grad)
    print(b.grad)