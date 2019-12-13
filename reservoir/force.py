# -*- coding: utf-8 -*-

# refer https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/

import numpy as np
from .esn import EchoStateNetwork


class FORCE(EchoStateNetwork):
    def __init__(self, in_size, reservoir_size, out_size, in_coef=0.1, res_coef=0.9999, fb_coef=0.01, in_dist="uniform", res_dist="normal", fb_dist="uniform"):
        super(FORCE, self).__init__(in_size, reservoir_size, out_size, in_coef, res_coef, fb_coef, in_dist, res_dist, fb_dist)
        

    def reset(self, forget_coef=0.1):
        super(FORCE, self).reset()

        # _r (r, r)
        self._r = (1 / forget_coef) * np.eye(self._w_r.shape[0])


    def update(self, t, mu=1.0):
        # t  (1, o): supervisor at the time
        # mu: coefficient

        # _r (r, r)
        self._r = (1 / mu) * (self._r - (self._r.dot(self._x.T).dot(self._x).dot(self._r.T)) / (mu + self._x.dot(self._r.T).dot(self._x.T)))

        # e  (1, o)
        e = self._y - t

        # _w_o (o, r)
        self._w_o = self._w_o - self._r.dot(self._x.T).dot(e).T