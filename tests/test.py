from steps import step0
import numpy as np
import unittest


# 单元测试
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = step0.Variable(np.array(3.0))
        y = step0.square(x)
        expected = np.array(9.0)
        self.assertEqual(expected, y.data)

    def test_backward(self):
        x = step0.Variable(np.random.rand(1))
        y = step0.square(x)
        y.backward()
        expected = step0.numeric_diff(step0.square, x)
        flag = np.allclose(x.grad, expected)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
