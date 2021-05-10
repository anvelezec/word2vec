"""
Layers to build models

Backpropagation guide: http://cs229.stanford.edu/notes2020spring/cs229-notes-deep_learning.pdf
"""

import numpy as np


class Linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = np.random.random((self.in_dim, self.out_dim))
        self.b = np.random.random(self.out_dim)
        self.x, self.dw, self.db = None, None, None

    def forward(self, x):
        self.x = x
        output = np.matmul(self.x, self.w) + self.b
        return output

    def backward(self, dx):
        s, dw = dx
        self.dw = np.matmul(self.x.T, s)
        self.db = np.sum(s, axis=0)
        s = np.matmul(s, self.w.T)
        return s, dw

    def sgd_step(self, lr):
        self.w -= self.dw * lr
        self.b -= self.db * lr


class Softmax:
    def __init__(self):
        self.x = None

    def forward(self, x):
        num = np.exp(x)
        self.x = num / np.expand_dims(np.sum(num, axis=1), axis=1)
        return self.x

    def backward(self, dx):
        # dx = self.x - y_true
        # dx = self.forward(self.x) * (1 - self.forward(self.x))
        # return dx
        pass
