import matplotlib.pyplot as plt
import numpy as np
import time
import timeit

import tensorflow as tf

from components.matting_v2 import MattingLaplacian as ML2
from components.matting_v3 import MattingLaplacian as ML3

eps, r = 1e-5, 1
n_iters = 10
iters = np.arange(n_iters)
repeats = 5

times2 = np.zeros((n_iters, repeats))
times3 = np.zeros((n_iters, repeats))
ticks = []

for i in iters:
    H, W = 50*(i+1), 50*(i+1)
    tick = '{}x{}'.format(H,W)
    ticks.append(tick)
    msg = '[{}/{}] Evaluating preprocessing time with image of size {}...'.format(i+1, n_iters, tick)
    img = tf.random.uniform((H,W,3))
    print(msg + ' (1/2)', end='\r')
    times2[i] = timeit.repeat(lambda: ML2(img, epsilon=eps, window_radius=r), repeat=5, number=1)
    print(msg + ' (2/2)', end='\r')
    times3[i] = timeit.repeat(lambda: ML3(img, epsilon=eps, window_radius=r), repeat=5, number=1)
    print(msg + ' Done.')

mean2, std2 = np.mean(times2, axis=1), np.std(times2, axis=1)
mean3, std3 = np.mean(times3, axis=1), np.std(times3, axis=1)
bound2, bound3 = 2*std2/np.sqrt(repeats), 2*std3/np.sqrt(repeats)

plt.plot(iters, mean2, c='b', label='Linear operator')
plt.fill_between(iters, mean2 + bound2, mean2 - bound2, alpha=0.15)
plt.plot(iters, mean3, c='r', label='Matrix')
plt.fill_between(iters, mean3 + bound3, mean3 - bound3, alpha=0.15)
plt.xlabel('Image size')
plt.ylabel('Preprocessing time (s)')
plt.xticks(iters, ticks, rotation=45, fontsize=8)
plt.grid(True)
plt.legend()
plt.show()
