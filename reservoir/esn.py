# -*- coding: utf-8 -*-

import numpy as np


class EchoStateNetwork(object):
    def __init__(self, in_size, reservoir_size, out_size, in_coef=0.1, res_coef=0.9999, fb_coef=0.01, in_dist="uniform", res_dist="normal", fb_dist="uniform"):
        # _w_i (r, i): weight of input [-in_coef in_coef]
        self._w_i = np.random.uniform(-in_coef, in_coef, (reservoir_size, in_size)).astype(np.float32) if in_dist == "uniform" else (np.random.randint(0, 2, (reservoir_size, in_size)).astype(np.float32) * 2 - 1) * in_coef

        # _w_r (r, r): weight of reservoir: init with random, divide by spectral radius [-res_coef res_coef]
        self._w_r = np.random.uniform(-1, 1, (reservoir_size, reservoir_size)).astype(np.float32) if res_dist == "uniform" else np.random.normal(0, 1, (reservoir_size, reservoir_size)).astype(np.float32)
        self._w_r = self._w_r / max(abs(np.linalg.eig(self._w_r)[0])) * res_coef

        # _w_b (r, o): weight of feedback [-fb_coef fb_coef]
        self._w_b = np.random.uniform(-fb_coef, fb_coef, (reservoir_size, out_size)).astype(np.float32) if fb_dist == "uniform" else (np.random.randint(0, 2, (reservoir_size, out_size)).astype(np.float32) * 2 - 1) * fb_coef

        # _w_o (o, r): weight of output (learnable): init with random
        self._w_o = np.random.rand(out_size, reservoir_size).astype(np.float32)

        self.reset()
        

    def reset(self):
        # _x (1, r): state of reservoir
        self._x = np.zeros((1, self._w_r.shape[0]), dtype=np.float32)

        # _x (N, r): state of reservoir through time N
        self._x_log = self._x

        # _y (1, o): output of network
        self._y = np.zeros((1, self._w_o.shape[0]), dtype=np.float32)


    def reset_except_log(self):
        # _x (1, r): state of reservoir
        self._x = np.zeros((1, self._w_r.shape[0]), dtype=np.float32)

        # _y (1, o): output of network
        self._y = np.zeros((1, self._w_o.shape[0]), dtype=np.float32)


    def __call__(self, u=None, leak=0.1):
        # u (1, i): input at a moment
        # leak: leak rate

        if not u is None:
            self._x = np.tanh((1 - leak) * self._x + leak * (u.dot(self._w_i.T) + self._x.dot(self._w_r.T) + self._y.dot(self._w_b.T)), dtype=np.float32)
        else:
            self._x = np.tanh((1 - leak) * self._x + leak * (self._x.dot(self._w_r.T) + self._y.dot(self._w_b.T)), dtype=np.float32)

        self._x_log = np.append(self._x_log, self._x, axis=0)

        self._y = self._x.dot(self._w_o.T)
        self._y = np.tanh(self._y, dtype=np.float32)
        return self._y


    def update(self, t, la=1.0):
        # t (N, o): supervisor through time N
        # la: coeffiicient of norm term

        self._w_o = np.linalg.inv((self._x_log[1:].T.dot(self._x_log[1:]) + la * np.eye(self._x_log[1:].shape[1]))).dot(self._x_log[1:].T).dot(t).T

    
    def update_descent(self, t, lr, la=1.0):
        # t (N, o): supervisor through time N
        # lr: learning rate
        # la: coeffiicient of norm term

        obj_w_o = np.linalg.inv((self._x_log[1:].T.dot(self._x_log[1:]) + la * np.eye(self._x_log[1:].shape[1]))).dot(self._x_log[1:].T).dot(t).T
        self._w_o = self._w_o + lr * (obj_w_o - self._w_o)


    def update_online(self, t, la=1.0):
        # t (1, o): supervisor at a moment
        # la: coeffiicient of norm term

        self._w_o = np.linalg.inv((self._x.T.dot(self._x) + la * np.eye(self._x.shape[1]))).dot(self._x.T).dot(t).T


    def update_online_descent(self, t, lr, la=1.0):
        # t (1, o): supervisor at a moment
        # lr: learning rate
        # la: coeffiicient of norm term

        obj_w_o = np.linalg.inv((self._x.T.dot(self._x) + la * np.eye(self._x.shape[1]))).dot(self._x.T).dot(t).T
        self._w_o = self._w_o + lr * (obj_w_o - self._w_o)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test data preparation
    LEN = 100
    sin0 = np.sin(np.linspace(0, 6.28, LEN)).reshape(-1, 1)
    sin1 = np.sin(np.linspace(1, 7.28, LEN)).reshape(-1, 1)
    sins = np.append(sin0, sin1, axis=1)

    # data plot
    us = sins[:-1]
    ys_target = sins[1:]
    plt.plot(us, label="u")
    plt.plot(ys_target, label="y_target")
    plt.legend()
    plt.show()

    # instantiation of ESN
    esn = EchoStateNetwork(in_size=2, reservoir_size=100, out_size=2)
    ys = []

    # before training
    esn.reset()
    for u in us:
        y = esn(u.reshape(1, 2), leak=0.1)
        ys.append(y)

    # training
    esn.update(ys_target, la=0.1)

    # after training
    esn.reset()
    for u in us:
        y = esn(u.reshape(1, 2), leak=0.1)
        ys.append(y)

    # data plot
    ys_target = np.tile(ys_target, (2, 1))
    plt.plot(ys_target, label="y_target")
    plt.plot(np.asarray(ys).reshape(-1, 2), label="y", linestyle='dashed')
    plt.legend()
    plt.show()