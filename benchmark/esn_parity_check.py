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
C_RES = 0.999
C_FB = 0.0
esn = EchoStateNetwork(1, RES_SIZE, 1, in_coef=C_IN, res_coef=C_RES, fb_coef=C_FB)


# input sequence
LEN = 20
us = np.random.choice([0, 1], LEN).astype(np.float32)

# target sequence
BIT = 5
ys_target = np.array([np.nan for i in range(BIT-1)] + [np.count_nonzero(us[i:i+BIT] == 1) % 2 for i in range(LEN-BIT+1)], dtype=np.float32)



# inputting
LEAK = 0.9
ys = []
for u in us:
    y = esn(u.reshape(1, 1), leak=LEAK)
    ys.append(y[0][0])

# compute loss
ys = np.asarray(ys)
loss0 = np.mean((ys[BIT-1:] - ys_target[BIT-1:]) ** 2)
print('loss before training', loss0)


# training
LA = 0.01
esn.update(ys_target[BIT-1:].reshape(-1, 1), t_start_at=BIT-1, la=LA)


# inputting again
ys = []
for u in us:
    y = esn(u.reshape(1, 1), leak=LEAK)
    ys.append(y[0][0])

# compute loss
ys = np.asarray(ys)
loss1 = np.mean((ys[BIT-1:] - ys_target[BIT-1:]) ** 2)
print('loss after training', loss1)


# plot
plt.plot(ys[BIT-1:], label='prediction', color='r', marker='o')
plt.plot(ys_target[BIT-1:], label='target', color='k', marker='o')
plt.legend()
plt.show()