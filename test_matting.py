import numpy as np
import time
import tensorflow as tf

from components.matting_v2 import MattingLaplacian
from components.matting_v3 import MattingLaplacian as ML3

H, W, C = 3,2,3
image = tf.random.uniform((H,W,C))
# image = tf.constant([
#     [[2,2],[0,0],[1,1]],
#     [[0,0],[0,0],[2,2]],
#     [[0,0],[0,0],[0,0]]
# ], dtype=tf.float32)
# image = tf.expand_dims(tf.constant([
#     [2,0,1],
#     [0,0,2],
#     [0,0,0]
# ], dtype=tf.float32), -1)
eps, r = 1e-5, 1
# m1 = MattingLaplacian(image, epsilon=eps, window_radius=r)
p = tf.reshape(image, (H*W,C))
t = [time.time()]
m2 = MattingLaplacian(image, epsilon=eps, window_radius=r)
t.append(time.time())
m3 = ML3(image, epsilon=eps, window_radius=r)
t.append(time.time())
Lp = m2.matmul(p)
t.append(time.time())

t = np.array(t)
print(*(t[1:] - t[:-1]))
# L = m2.matmul(cols)
# L1 = m2._matmul1(cols)
# print()
# tf.print(tf.squeeze(L))
# print()
# tf.print(tf.squeeze(L1))
# tf.print(L1)
# print()
# t1 = time.time()
L = m2.to_dense()
L3 = m3.to_dense()
# t2 = time.time()
# print(t2-t1)
tf.print(L)
tf.print(L3)
tf.print(tf.reduce_min(tf.linalg.eigvalsh(L)))
# tf.print(m2.matmul(cols[:,0]))
# tf.print(m2.matmul(cols[:,1]))
