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
C_IN = 1.0
C_RES = 0.999
C_FB = 0.0
esn = EchoStateNetwork(1, RES_SIZE, 1, in_coef=C_IN, res_coef=C_RES, fb_coef=C_FB)


# input sequence
LEN = 20
us = np.random.choice([0, 1], LEN).astype(np.float32)

# target sequence
DELAY = 5
ys_target = np.array([np.nan for i in range(DELAY)] + [us[i] for i in range(LEN-DELAY)], dtype=np.float32)



# inputting
LEAK = 0.9
ys = []
for u in us:
    y = esn(u.reshape(1, 1), leak=LEAK)
    ys.append(y[0][0])

# compute loss
ys = np.asarray(ys)
loss0 = np.mean((ys[DELAY:] - ys_target[DELAY:]) ** 2)
print('loss before training', loss0)


# training
LA = 0.01
esn.update(ys_target[DELAY:].reshape(-1, 1), t_start_at=DELAY, la=LA)


# inputting again
ys = []
for u in us:
    y = esn(u.reshape(1, 1), leak=LEAK)
    ys.append(y[0][0])

# compute loss
ys = np.asarray(ys)
loss1 = np.mean((ys[DELAY:] - ys_target[DELAY:]) ** 2)
print('loss after training', loss1)


# plot
plt.plot(ys[DELAY:], label='prediction', color='r', marker='o')
plt.plot(ys_target[DELAY:], label='target', color='k', marker='o')
plt.legend()
plt.show()