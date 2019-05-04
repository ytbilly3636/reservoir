# -*- coding: utf-8 -*-

import numpy as np

from esn import EchoStateNetwork


class PredictiveCoding(EchoStateNetwork):
    def __init__(self, reservoir_size, out_size):
        # _w_r (r, r): weight of reservoir: init with random, divide by spectral radius
        self._w_r = np.random.normal(0, 1, (reservoir_size, reservoir_size)).astype(np.float32)
        self._w_r = self._w_r / max(abs(np.linalg.eig(self._w_r)[0]))

        # _w_b (r, o)
        self._w_b = np.random.rand(reservoir_size, out_size).astype(np.float32)

        # _w_o (o, r): weight of output (learnable): init with random
        self._w_o = np.random.rand(out_size, reservoir_size).astype(np.float32)

        self.reset()


    def reset(self):
        self._x = np.zeros((1, self._w_r.shape[0]), dtype=np.float32)
        self._y = np.zeros((1, self._w_o.shape[0]), dtype=np.float32)


    #def reset_log(self):
    #    self._x_log = np.zeros((1, self._w_r.shape[0]), dtype=np.float32)


    def __call__(self, d=None, leak_x=0.25, leak_r=1.0):
        if d == None:
            self._x = (1 - leak_x) * self._x + np.tanh(self._x.dot(self._w_r.T) + self._y.dot(self._w_b.T))
        else:
            r = d - self._y
            self._x = (1 - leak_x) * self._x + np.tanh(self._x.dot(self._w_r.T) + (self._y + leak_r * r).dot(self._w_b.T))

        #self._x_log = np.append(self._x_log, self._x, axis=0)
        self._y = np.tanh(self._x.dot(self._w_o.T))
        return self._y


    def update(self, t, lr=0.1):
        self._w_o = np.linalg.inv((self._x.T.dot(self._x) + lr * np.eye(self._x.shape[1]))).dot(self._x.T).dot(t).T


if __name__ == '__main__':
    pc = PredictiveCoding(reservoir_size=100, out_size=1)
    pc.reset()

    t = np.arange(0, 6.3, 0.1)
    input = np.sin(t).reshape(-1, 1)

    import six
    import matplotlib.pyplot as plt

    # train
    ys = []
    for epoch in six.moves.range(10):
        for i in input:
            ys.append(pc(i.reshape(1, 1)))
            pc.update(i.reshape(1, 1))

    # test
    for i in six.moves.range(63*10):
        ys.append(pc())

    # plot
    plt.plot(np.asarray(ys).reshape(-1, 1), label="y")
    plt.legend()
    plt.show()
