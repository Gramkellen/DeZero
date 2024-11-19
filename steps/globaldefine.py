import numpy as np


# ----------------- Function Tools ----------------------
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# 中心差分函数,对函数 f 在 x 处进行求导
def numeric_diff(f, x, esp=1e-4):
    x0 = Variable(x.data + esp)
    x1 = Variable(x.data - esp)
    return (f(x0).data - f(x1).data) / (2.0 * esp)


# --------------Global Variable and Functions -------------------
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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            # 确保元素为元组
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                if x.creator is not None:
                    functions.append(x.creator)
# 400 1 100 1

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)   # 使用星号解包，不使用一个列表的传递方式
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs  # 正向传播保存值
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    # 正向传播
    def forward(self, inputs):
        raise NotImplementedError

    # 反向传播
    def backward(self, grad):
        raise NotImplementedError
