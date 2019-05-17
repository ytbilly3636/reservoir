# -*- coding: utf-8 -*-

import numpy as np


class PredictiveCoding():
    def __init__(self, reservoir_size, out_size):
        # _w_r (r, r): weight of reservoir: init with random, divide by spectral radius
        self._w_r = np.random.normal(0, 1, (reservoir_size, reservoir_size)).astype(np.float32)
        self._w_r = self._w_r / max(abs(np.linalg.eig(self._w_r)[0]))

        # _w_b (r, o)
        #self._w_b = np.random.rand(reservoir_size, out_size).astype(np.float32)
        #self._w_b = (np.random.randint(0, 2, (reservoir_size, out_size)).astype(np.float32) * 2 - 1) * 0.1
        self._w_b = np.ones((reservoir_size, out_size), dtype=np.float32) * 0.1

        # _w_o (o, r): weight of output (learnable): init with random
        self._w_o = np.random.rand(out_size, reservoir_size).astype(np.float32)

        self.reset()


    def reset(self):
        # _x (1, r): state of reservoir
        self._x = np.zeros((1, self._w_r.shape[0]), dtype=np.float32)

        # _y (1, o): output of reservoir
        self._y = np.zeros((1, self._w_o.shape[0]), dtype=np.float32)

        # _x_log (N, r): state of reservoir through time N
        self._x_log = self._x

        # _y_log (N, o): output of reservoir through time N
        self._d_log = self._y


    def __call__(self, d=None, attention=1.0, leak=0.25):
        # d (1, o): sensory input
        # r (1, o): prediction error
        r = d - self._y if not d is None else - self._y
        self._d_log = np.append(self._d_log, d, axis=0) if not d is None else np.append(self._d_log, np.zeros((1, self._w_o.shape[0]), dtype=np.float32), axis=0)

        self._x = self._x - leak * self._x + np.tanh(self._x.dot(self._w_r.T) + (self._y + attention * r).dot(self._w_b.T))
        self._x_log = np.append(self._x_log, self._x, axis=0)

        self._y = np.tanh(self._x.dot(self._w_o.T))
        return self._y


    def update(self, sparseness=0.1):
        self._w_o = np.linalg.inv((self._x_log[1:].T.dot(self._x_log[1:]) + sparseness * np.eye(self._x_log[1:].shape[1]))).dot(self._x_log[1:].T).dot(self._d_log[1:]).T


    def update_descent(self, lr, sparseness=0.1):
        obj_w_o = np.linalg.inv((self._x_log[1:].T.dot(self._x_log[1:]) + sparseness * np.eye(self._x_log[1:].shape[1]))).dot(self._x_log[1:].T).dot(self._d_log[1:]).T
        self._w_o = self._w_o + lr * (obj_w_o - self._w_o)


if __name__ == '__main__':
    pc = PredictiveCoding(reservoir_size=500, out_size=1)
    pc.reset()

    t = np.arange(0, 6.3, 0.1)
    input = np.sin(t).reshape(-1, 1)

    import six
    import matplotlib.pyplot as plt

    xs = []
    ys = []

    # train
    for i in input:
        xs.append(i)
        ys.append(pc(i.reshape(1, 1), attention=0.5, leak=0.1))
    pc.update(sparseness=0.1)

    # test
    for i, sin in enumerate(input):
        attention = 1.0 if i < len(t) / 2 else 0.0
        ys.append(pc(sin.reshape(1, 1), attention=attention))

    # plot
    plt.plot(np.asarray(xs).reshape(-1, 1), label="input")
    plt.plot(np.asarray(ys).reshape(-1, 1), label="output")
    plt.legend()
    plt.show()
