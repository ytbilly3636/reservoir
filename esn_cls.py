# -*- coding: utf-8 -*-

import numpy as np

from esn import EchoStateNetwork


class EchoStateNetworkClassifier(EchoStateNetwork):
    def reset(self):
        # _x (1, r): state of reservoir
        self._x = np.zeros((1, self._w_r.shape[0]), dtype=np.float32)


    def __call__(self, u, leak=0.1):
        # u (1, i): input at a moment

        self._x = np.tanh((1 - leak) * self._x + leak * (u.dot(self._w_i.T) + self._x.dot(self._w_r.T)))
        return np.tanh(self._x.dot(self._w_o.T))


    def update(self, t, lr=0.01):
        # t (1, o): supervisor at N

        self._w_o = np.linalg.inv((self._x.T.dot(self._x) + lr * np.eye(self._x.shape[1]))).dot(self._x.T).dot(t).T


if __name__ == '__main__':
    esn = EchoStateNetworkClassifier(in_size=1, reservoir_size=100, out_size=2)

    t = np.arange(0, 15, 0.1)
    inputs = np.sin(t).reshape(-1, 1, 1)
    supers = np.eye(2)

    import six
    import matplotlib.pyplot as plt

    esn.reset()
    ys = []
    for i in inputs:
        ys.append(esn(i))
        if i[0][0] > 0:
            esn.update(supers[0].reshape(1, -1))
        else:
            esn.update(supers[1].reshape(1, -1))

    plt.plot(np.asarray(ys).reshape(150, 2)[:, 0], label="out0")
    plt.plot(np.asarray(ys).reshape(150, 2)[:, 1], label="out1")
    plt.plot(inputs[:, 0], label="input")
    plt.legend()
    plt.show()
