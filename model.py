"""
Skipgram model definition
"""

from layers import Linear, Softmax


class Skipgram:
    def __init__(self, params):
        self.l1 = Linear(in_dim=params["l1"]["in_dim"], out_dim=params["l1"]["out_dim"])
        self.l2 = Linear(in_dim=params["l2"]["in_dim"], out_dim=params["l2"]["out_dim"])
        self.softmax = Softmax()

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.l2.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self, dx):
        # dx = self.softmax.backward(dx)
        dx = self.l2.backward(dx)
        dx = self.l1.backward(dx)

    def update(self, lr):
        self.l1.sgd_step(lr)
        self.l2.sgd_step(lr)
