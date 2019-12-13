# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# add path of reservoir and import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reservoir import EchoStateNetwork


# reservoir
RES_SIZE = 50
C_IN = 0.5
C_RES = 0.999
C_FB = 0.0
esn = EchoStateNetwork(1, RES_SIZE, 1, in_coef=C_IN, res_coef=C_RES, fb_coef=C_FB)


# input sequence
LEN = 20
us = np.random.uniform(-1, 1, LEN).astype(np.float32)

# target sequence (NARMA2)
ALPHA = 0.4
BETA = 0.4
GAMMA = 0.6
DELTA = 0.1
ys_target = np.zeros(us.shape, us.dtype)
for i in range(2, us.shape[0]):
    ys_target[i] = ALPHA * ys_target[i-1] + BETA * ys_target[i-1] * ys_target[i-2] + GAMMA * (us[i] ** 3) + DELTA


# inputting
LEAK = 0.9
ys = []
for u in us:
    y = esn(u.reshape(1, 1), leak=LEAK)
    ys.append(y[0][0])

# compute loss
ys = np.asarray(ys)
loss0 = np.mean((ys - ys_target) ** 2)
print('loss before training', loss0)


# training
LA = 0.01
esn.update(ys_target.reshape(-1, 1), la=LA)


# inputting again
ys = []
for u in us:
    y = esn(u.reshape(1, 1), leak=LEAK)
    ys.append(y[0][0])

# compute loss
ys = np.asarray(ys)
loss1 = np.mean((ys - ys_target) ** 2)
print('loss after training', loss1)


# plot
plt.plot(ys, label='prediction', color='r', marker='o')
plt.plot(ys_target, label='target', color='k', marker='o')
plt.legend()
plt.show()