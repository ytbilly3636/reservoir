# prefrontal_cortex

## What is this
This repository is for developing reservoir computing models (echo state networks).  
Especially, the models are used for prefrontal cortex models.

## Requirement
* Python
* NumPy
* Matplotlib

## Quick Start
To execute test script of echo state network,

```sh
$ python reservoir/esn.py
```

To execute benchmark test of short term memory using echo state network,

```sh
$ python benchmark/esn_short_term_memory.py
```

## How to use reservoir
#### Echo State Network
Import an "EchoStateNetwork" class in the "reservoir",

```python
# append "reservoir" to sys.path if you need
import sys
import os
sys.path.append('PATH TO reservoir/')

from reservoir import EchoStateNetwork
```

Instantiate an object from the "EchoStateNetwork",

```python
IN_SIZE = 1
RES_SIZE = 100
OUT_SIZE = 1

esn = EchoStateNetwork(IN_SIZE, RES_SIZE, OUT_SIZE)
```

Execute \_\_call\_\_() function of the object to run forward propagation at a moment,

```python
import numpy as np

# input sequence "us" is a sinusoidal
# target sequence "ys_target" is the same sinusoidal except shifted phase
sinusoidal = np.sin(np.linspace(0, np.pi*2, 100)).reshape(-1, 1)
us = sinusoidal[:-1]
ys_target = sinusoidal[1:]

# reset the esn before inputting
esn.reset()

# forward propagation
# the function receives (1, IN_SIZE) array
# argument "leak" means a leak rate of the reservoir
LEAK = 0.1
for u in us:
    esn(u.reshape(1, 1), leak=LEAK)
```

Execute update() function of the object to run ridge regression after the forward propagation,

```python
# the function receives (N, OUT_SIZE) array (N is the length of the sequence)
# argument "la" means a coeffiicient of norm term of the ridge regression
LA = 1.0
esn.update(ys_target, la=LA)
```

After the ridge regression, you can check how the esn adapts target sequence,

```python
import matplotlib.pyplot as plt

# reset the esn before inputting
esn.reset()

# forward propagtation again
ys = []
for u in us:
    y = esn(u.reshape(1, 1), leak=LEAK)
    ys.append(y[0][0])

# plot the prediction and the target
ys = np.asarray(ys)
plt.plot(ys, label='prediction')
plt.plot(ys_target.reshape(-1, ), label='target')
plt.legend()
plt.show()
```

#### FORCE
Import a "FORCE" class in the "reservoir",

```python
from reservoir import FORCE
```

Instantiate an object from the "FORCE",

```python
IN_SIZE = 1
RES_SIZE = 100
OUT_SIZE = 1

force = FORCE(IN_SIZE, RES_SIZE, OUT_SIZE)
```

Execute \_\_call\_\_() function of the object to run forward propagation at a moment,
execute update() function of the object to train the network as the same time since online learning,

```python
# reset the force before inputting
force.reset()

# forward propagation and training
# __call__() receives (1, IN_SIZE) array
# update() receives (1, OUT_SIZE) array
# argument "leak" means a leak rate of the reservoir
LEAK = 0.1
for u, y_target in zip(us, ys_target):
    force(u.reshape(1, 1), leak=LEAK)
    force.update(y_target.reshape(1, 1))
```

---

scripted by  
Yuichiro Tanaka
tanaka.yuichiro483@mail.kyutech.jp  
Tamukoh Laboratory, Kyushu Institute of Technology