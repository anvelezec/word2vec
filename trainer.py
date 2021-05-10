"""
Training step
"""
import numpy as np

from model import Skipgram
from losses import Crossentropy


sentence = "hello i like to ride bicycle"
x = [
    ([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]),
    ([0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
    ([0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]),
    ([0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]),
    ([0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]),
    ([0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]),
    ([0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]),
    ([0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]),
    ([0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]),
    ([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]),
]

emb_dim = 3
voc_size = len(x[0][0])

params = {
    "l1": {"in_dim": voc_size, "out_dim": emb_dim},
    "l2": {"in_dim": emb_dim, "out_dim": voc_size},
}

feature = [item[0] for item in x]
labels = [item[1] for item in x]

feature, labels, dx, y_true = np.array(feature), np.array(labels), None, None

# Model instantiation
net = Skipgram(params=params)
crossentropy = Crossentropy()
n_epochs = 1000

for epoch in range(n_epochs):
    # Forward pass
    y_pred = net.forward(feature)

    # Loss calculation
    loss = crossentropy.forward(labels, y_pred)
    print(loss)

    # Backward pass
    s = crossentropy.backward(labels, y_pred)
    net.backward((s, None))
    net.update(lr=0.01)
