# -*- coding: utf-8 -*-

import numpy as np
from .esn import EchoStateNetwork


class PredictiveCoding(object):
    def __init__(self, in_size, reservoir_size, in_coef=0.1, res_coef=0.9999, in_dist="uniform", res_dist="normal"):
        self._esn = EchoStateNetwork(in_size, reservoir_size, in_size, in_coef, res_coef, 0.0, in_dist, res_dist)
        self.reset()


    def reset(self):
        self._esn.reset()
        
        # _d_log (N, i): sensory input through time N
        self._d_log = np.zeros((1, self._esn._w_i.shape[1]), dtype=np.float32)


    def __call__(self, d=None, attention=1.0, leak=0.1):
        # d (1, i): sensory input
        # atttention: attention rate
        # leak: leak rate

        # prediction error
        if not d is None:
            r = d - self._esn._y
            self._d_log = np.append(self._d_log, d, axis=0)
        else:
            r = - self._esn._y

        # input of reservoir
        u = self._esn._y + attention * r
        y = self._esn(u, leak)

        return y


    def update(self, la=1.0):
        # la: coeffiicient of norm term

        self._esn.update(self._d_log[1:], la)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test data preparation
    LEN = 100
    sin = np.sin(np.linspace(0, 6.28, LEN)).reshape(-1, 1)

    # instantiation of PCRC
    pcrc = PredictiveCoding(in_size=1, reservoir_size=500)
    ys = []

    # before training
    pcrc.reset()
    for d in sin:
        y = pcrc(d.reshape(1, 1), attention=1.0, leak=0.1)
        ys.append(y)

    # training
    pcrc.update(la=1.0)

    # after training (error driven)
    pcrc.reset()
    for d in sin:
        y = pcrc(d.reshape(1, 1), attention=1.0, leak=0.1)
        ys.append(y)
    
    # after training (free run)
    for d in sin:
        y = pcrc(None, attention=0.0, leak=0.1)
        ys.append(y)

    # data plot
    plt.plot(np.asarray(ys).reshape(-1, 1))
    plt.show()
