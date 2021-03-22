import tensorflow as tf
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = tf.convert_to_tensor(a)
b[0] = [5, 6]
sess = tf.Session()
print(sess.run(b))
