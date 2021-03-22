import numpy as np
from PIL import Image
import os

x_test = np.load('data_np/x_test.npy')
y_test = np.load('data_np/y_test.npy')

for i in range(len(x_test)):
    img = Image.fromarray(x_test[i].astype('uint8'), 'RGB')
    img.save('images_raw' + os.path.sep + str(i) + '.bmp')

print(x_test)
print(y_test)
