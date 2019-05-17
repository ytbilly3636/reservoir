# -*- coding: utf-8 -*-

import numpy as np


class EchoStateNetwork(object):
    def __init__(self, in_size, reservoir_size, out_size):
        # _w_i (r, i): weight of input: init with -0.1 or 0.1
        self._w_i = (np.random.randint(0, 2, (reservoir_size, in_size)).astype(np.float32) * 2 - 1) * 0.1

        # _w_r (r, r): weight of reservoir: init with random, divide by spectral radius
        self._w_r = np.random.normal(0, 1, (reservoir_size, reservoir_size)).astype(np.float32)
        self._w_r = self._w_r / max(abs(np.linalg.eig(self._w_r)[0]))

        # _w_o (o, r): weight of output (learnable): init with random
        self._w_o = np.random.rand(out_size, reservoir_size).astype(np.float32)

        self.reset()
        

    def reset(self):
        # _x (1, r): state of reservoir
        self._x = np.zeros((1, self._w_r.shape[0]), dtype=np.float32)

        # _x (N, r): state of reservoir through time N
        self._x_log = self._x


    def __call__(self, u=None, leak=0.1):
        # u (1, i): input at a moment

        if not u is None:
            self._x = np.tanh((1 - leak) * self._x + leak * (u.dot(self._w_i.T) + self._x.dot(self._w_r.T)))
        else:
            self._x = np.tanh((1 - leak) * self._x + leak * self._x.dot(self._w_r.T))

        self._x_log = np.append(self._x_log, self._x, axis=0)
        return np.tanh(self._x.dot(self._w_o.T))


    def update(self, t, la=0.1):
        # t (N, o): supervisor through time N

        self._w_o = np.linalg.inv((self._x_log[1:].T.dot(self._x_log[1:]) + la * np.eye(self._x_log[1:].shape[1]))).dot(self._x_log[1:].T).dot(t).T

    
    def update_descent(self, t, lr, la=0.1):
        obj_w_o = np.linalg.inv((self._x_log[1:].T.dot(self._x_log[1:]) + la * np.eye(self._x_log[1:].shape[1]))).dot(self._x_log[1:].T).dot(t).T
        self._w_o = self._w_o + lr * (obj_w_o - self._w_o)


if __name__ == '__main__':
    esn = EchoStateNetwork(in_size=2, reservoir_size=100, out_size=2)

    t = np.arange(0, 100)
    inputs = np.append(np.sin(t).reshape(-1, 1), np.cos(t).reshape(-1, 1), axis=1)
    
    import six
    import matplotlib.pyplot as plt

    yss = []
    for epoch in six.moves.range(10):
        esn.reset()
        ys = []
        for i in t:
            ys.append(esn(inputs[i]))

        yss.append(ys)
        esn.update(inputs)

    plt.plot(np.asarray(yss[0]).reshape(100, 2)[:, 0], label="0")
    plt.plot(np.asarray(yss[9]).reshape(100, 2)[:, 0], label="9")
    plt.plot(inputs[:, 0], label="t")
    plt.legend()
    plt.show()