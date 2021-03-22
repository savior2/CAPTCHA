import keras
import numpy as np
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_milli_time = lambda: 0  # int(round(time.time() * 1000))


def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.3 * cross_ent + 0.7 * cross_ent_adv

    return adv_loss


def rotate(points, theta):
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), 0,
                                tf.sin(theta), tf.cos(theta), 0,
                                0, 0, 1])
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


def rotate_random(points, theta_range):
    theta = tf.random.uniform(shape=(), seed=current_milli_time(), minval=theta_range[0], maxval=theta_range[1],
                              dtype=tf.dtypes.float32)
    rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), 0,
                                tf.sin(theta), tf.cos(theta), 0,
                                0, 0, 1])
    rotation_matrix = tf.reshape(rotation_matrix, (3, 3))
    return tf.matmul(points, rotation_matrix)


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


def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_rotated_acc_metric(theta, model, fgsm, fgsm_params):
    def adv_rotated_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)

        x_adv = tf_rotate(x_adv)

        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    adv_rotated_acc.__name__ = 'rotated_{}'.format(theta)
    return adv_rotated_acc


def get_adversarial_random_loss(model, fgsm, fgsm_params, theta_range=(-45, 45), ratio_range=(0.8, 1.0)):
    def adv_random_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)

        x_adv = tf_rotate(x_adv)

        x_adv = resize_and_pad_random(x_adv, 48, 48, ratio_range)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)
        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.3 * cross_ent + 0.7 * cross_ent_adv

    return adv_random_loss


def get_adversarial_random_acc_metric(model, fgsm, fgsm_params, theta_range=(-45, 45), ratio_range=(0.8, 1.0)):
    def adv_random_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.get_input_at(0), **fgsm_params)
        x_adv = tf_rotate(x_adv)
        x_adv = resize_and_pad_random(x_adv, 48, 48, ratio_range)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)
        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_random_acc


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


num_classes = 10
batch_size = 32
epochs = 50

x_train = np.load('../data/data_np/x_train.npy')
y_train = np.load('../data/data_np/y_train.npy')
x_test = np.load('../data/data_np/x_test.npy')
y_test = np.load('../data/data_np/y_test.npy')

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model0 = keras.models.load_model('../captcha_model/resnet/resnet_captcha.h5')
fgsm_params = {'eps': 0.1,
               'clip_min': 0.,
               'clip_max': 1.}

model1 = keras.models.load_model('../captcha_model/resnet/resnet_captcha.h5')
"""keras.models.load_model('cifar_model/resnet/resnet_cifar_fgsm_adv_train_loss3pure7adv_puregenadv.h5',
                                 custom_objects={'adv_loss': 'categorical_crossentropy',
                                                 'adv_acc': 'accuracy'})
    #keras.models.load_model('cifar_model/resnet/resnet_cifar_pure.h5')"""

wrap = KerasModelWrapper(model0)
fgsm = FastGradientMethod(wrap)
adv_acc = get_adversarial_acc_metric(model1, fgsm, fgsm_params)
adv_random = get_adversarial_random_acc_metric(model1, fgsm, fgsm_params, (-45, 45), (0.8, 1.0))
metrics = ['accuracy', adv_acc, adv_random]
adv_loss = get_adversarial_random_loss(model1, fgsm,
                                       fgsm_params)  # get_adversarial_loss(model1, fgsm, fgsm_params) #get_adversarial_random_loss(model1, fgsm, fgsm_params)

model1.compile(
    optimizer=Adam(learning_rate=lr_schedule(0)),
    loss=adv_loss,
    metrics=metrics
)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

print('start')
# print(model1.evaluate(x_test, y_test, verbose=2))
# print(model1.metrics_names)
model1.fit(x_train, y_train,
           batch_size=batch_size,
           epochs=epochs,
           validation_data=(x_test, y_test),
           shuffle=True,
           verbose=1,
           callbacks=callbacks,
           validation_freq=5)

model1.save('../captcha_model/resnet/resnet_captcha_fgsm_adv_rotate_train.h5')

print(model1.evaluate(x_test, y_test, verbose=2))
print(model1.metrics_names)
