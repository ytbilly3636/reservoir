# -*- coding: utf-8 -*-

import numpy as np

from esn import EchoStateNetwork


class EchoStateNetworkClassifier(EchoStateNetwork):
    def update(self, t, la=1.0):
        # t (1, o): supervisor at N
        # la: coeffiicient of norm term

        self._w_o = np.linalg.inv((self._x.T.dot(self._x) + la * np.eye(self._x.shape[1]))).dot(self._x.T).dot(t).T


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
