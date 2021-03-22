import os

import numpy as np
from keras.models import load_model
import keras

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

test_x = np.load('../data/data_adv_np/x_test.npy')
test_x = test_x.astype('float32') / 255

test_y = np.load('../data/data_adv_np/y_test.npy')
test_y = keras.utils.to_categorical(test_y, 10)

model1 = load_model('../captcha_model/resnet/resnet_captcha_fgsm_adv_train2_eps01.h5',
                    custom_objects={'adv_random_loss': 'categorical_crossentropy',
                                    'adv_loss': 'categorical_crossentropy',
                                    'adv_random_acc': 'accuracy',
                                    'adv_acc': 'accuracy'})

model1.evaluate(test_x, test_y, verbose=1)
score = model1.evaluate(test_x, test_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# result = [0 for i in range(10)]
# preds = model1.predict(test_x)
# index = np.argmax(preds, axis=1)
# print(index)
# for i in index:
#     result[i] = result[i] + 1
# print(result)
