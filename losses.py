"""
Losses definition
"""
import numpy as np


class Crossentropy:
    def forward(self, y_true, y_pred):
        loss = np.sum(-1 * y_true * np.log(y_pred)) / y_pred.shape[0]
        return loss

    def backward(self, y_true, y_pred):
        dx = y_pred - y_true
        return dx
