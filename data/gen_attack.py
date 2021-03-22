import os

import keras
import numpy as np
from PIL import Image
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model0 = keras.models.load_model('../captcha_model/resnet/resnet_captcha.h5')

wrap = KerasModelWrapper(model0)
fgsm = FastGradientMethod(wrap, sess=K.get_session())
fgsm_params = {'eps': 0.1,
               'clip_min': 0.,
               'clip_max': 1.}

test_x = np.load('data_adv_np/x_test.npy')
test_x = test_x

test_x = test_x.astype('float32') / 255

for i in range(100):
    img_sample = Image.fromarray((test_x[i] * 255).astype('uint8'), 'RGB')
    img_sample.save(os.path.join('images_raw', 'raw_' + str(i) + '.bmp'))

print(test_x.shape)
x = fgsm.generate_np(test_x, **fgsm_params)
for i in range(100):
    img_adv_sample = Image.fromarray((x[i] * 255).astype('uint8'), 'RGB')
    img_adv_sample.save(os.path.join('images_adv', 'adv_' + str(i) + '.bmp'))
np.save('data_adv_np/x_test_adv', x)
print('end')

"""model1 = keras.models.load_model('cifar_model/resnet/resnet_cifar_fgsm_adv_train_loss3pure7adv_puregenadv.h5',
                                 custom_objects={'adv_loss': 'categorical_crossentropy',
                                                 'adv_acc': 'accuracy'})
adv_acc = get_adversarial_acc_metric(model1, fgsm, fgsm_params)
adv_random = get_adversarial_random_acc_metric(model1, fgsm, fgsm_params, (-45, 45), (0.6, 1.0))
metrics = ['accuracy', adv_acc, adv_random]
adv_loss = get_adversarial_loss(model1, fgsm, fgsm_params)
model1.compile(
    optimizer=Adam(learning_rate=lr_schedule(0)),
    loss=adv_loss,
    metrics=metrics
)
print(model1.evaluate(x_test, y_test, verbose=1))
print(model1.metrics_names)"""
