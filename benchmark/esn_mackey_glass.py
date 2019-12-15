# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# add path of reservoir and import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reservoir import EchoStateNetwork


# reservoir
RES_SIZE = 100
C_IN = 1.0
C_RES = 0.9
C_FB = 0.0
esn = EchoStateNetwork(1, RES_SIZE, 1, in_coef=C_IN, res_coef=C_RES, fb_coef=C_FB)


# Mackey-Glass
LEN = 50
BETA = 2.0
GAMMA = 1.0
TAU = 2
N = 9.65
mackey_glass = np.random.rand(LEN, ).astype(np.float32)
for i in range(TAU, LEN):
    mackey_glass[i] = (1.0 - GAMMA) * mackey_glass[i-1] + BETA * mackey_glass[i-TAU] / (1.0 + pow(mackey_glass[i-TAU], N))

# normaliation because of output range of esn
mackey_glass = mackey_glass / np.max(mackey_glass)


# input sequence
DELAY = 1
us = mackey_glass[:-DELAY]

# target sequence
ys_target = mackey_glass[DELAY:]


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
LA = 0.001
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