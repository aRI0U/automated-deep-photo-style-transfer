import numpy as np
import tensorflow as tf

from components.matting_v3 import MattingLaplacian
from components.matting_v2 import MattingLaplacian2

H, W, C = 3, 3, 2
# image = tf.random.uniform((H,W,C))
image = tf.constant([
    [[2,2],[0,0],[1,1]],
    [[0,0],[0,0],[2,2]],
    [[0,0],[0,0],[0,0]]
], dtype=tf.float32)
eps, r = 1e-2, 1
m1 = MattingLaplacian(image, epsilon=eps, window_radius=r)
m2 = MattingLaplacian2(image, epsilon=eps, window_radius=r)

cols = np.zeros((H*W,2))
cols[0,0], cols[1,1] = 1, 1
cols = tf.constant(cols, dtype=image.dtype)

L = m2.matmul(cols)
# tf.print(L1)
# print()
# L = m2.to_dense()
print()
tf.print(L)
# tf.print([L[i,i] for i in range(H*W)])
# tf.print(tf.linalg.det(L))
# tf.print(m2.matmul(cols[:,0]))
# tf.print(m2.matmul(cols[:,1]))
bimage = m2.add_border(image)[1:,1:]

means = np.zeros((H,W,C))

for i in range(H):
    for j in range(W):
        means[i,j] = np.mean(bimage[i:i+2*r+1,j:j+2*r+1])

covs = np.zeros((H,W,C))

for i in range(H):
    for j in range(W):
        covs[i,j] = np.mean(bimage[i:i+2*r+1,j:j+2*r+1]**2) - means[i,j]**2

delta_inv = np.zeros((H,W,C))

for i in range(H):
    for j in range(W):
        delta_inv[i,j] = 1/(covs[i,j] + eps/(2*r+1)**2)
