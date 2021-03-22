import tensorflow as tf
import numpy as np
from keras import backend as K
from PIL import Image
import os
import time

current_milli_time = lambda: 0


def rotate(points, theta):
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), 0,
                                tf.sin(theta), tf.cos(theta), 0,
                                0, 0, 1])
    print(rotation_matrix)
    rotation_matrix = tf.reshape(rotation_matrix, (3, 3))
    return tf.matmul(points, rotation_matrix)


def tf_rotate(input_image, min_angle=-np.pi / 4, max_angle=np.pi / 4):
    '''
    TensorFlow对图像进行随机旋转
    :param input_image: 图像输入
    :param min_angle: 最小旋转角度
    :param max_angle: 最大旋转角度
    :return: 旋转后的图像
    '''
    # distorted_image = tf.expand_dims(input_image, 0)
    distorted_image = input_image
    random_angles = tf.random.uniform(shape=(tf.shape(distorted_image)[0],), minval=min_angle, maxval=max_angle)
    distorted_image = tf.contrib.image.transform(
        distorted_image,
        tf.contrib.image.angles_to_projective_transforms(
            random_angles,
            tf.cast(tf.shape(distorted_image)[1], tf.float32),
            tf.cast(tf.shape(distorted_image)[2], tf.float32)
        ))
    # rotate_image = tf.squeeze(distorted_image, [0])
    rotate_image = distorted_image
    return rotate_image


def resize_and_pad_random(points, image_height, image_width, ratio_range=(0.8, 1.0)):
    new_height = tf.cast(
        tf.random.uniform(shape=(), seed=current_milli_time(), minval=image_height * ratio_range[0],
                          maxval=image_height * ratio_range[1]),
        dtype=tf.dtypes.int32)
    new_width = tf.cast(
        tf.random.uniform(shape=(), seed=current_milli_time(), minval=image_width * ratio_range[0],
                          maxval=image_width * ratio_range[1]),
        dtype=tf.dtypes.int32)
    points = tf.image.resize(points, tf.stack([new_height, new_width]))
    points = tf.image.pad_to_bounding_box(points, 48 - new_height, 48 - new_width, 48, 48)
    return points


x_test = np.load('data_adv_np/x_test_adv.npy').astype('float32')
x = tf_rotate(x_test)
x = resize_and_pad_random(x, 48, 48)
x_test = K.eval(x)
for i in range(10):
    # x = tf_rotate(x_test[i])
    # x = resize_and_pad_random(x, 48, 48)
    # x = K.eval(x)
    image = Image.fromarray((x_test[i] * 255).astype('uint8'), 'RGB')
    image.save('images_raw/test.bmp')
    # x_test[i] = x
    # print(i)

np.save('data_adv_np/x_test_adv_rotate', x_test)
