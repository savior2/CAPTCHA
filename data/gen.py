# captcha 0.3
from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import os


def gen_captcha(text, width, height):
    # 定义图片对象
    image_captcha = ImageCaptcha(width=width, height=height)
    # 生成图像
    image = Image.open(image_captcha.generate(text))
    return text, image


def save_image(image, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    image.save(path + os.path.sep + filename)


def get_np(vocab, size, width, height, channel=3):
    x = np.zeros((size * len(vocab), width, height, channel))
    y = np.zeros((size * len(vocab), 1))
    k = 0
    for i in range(len(vocab)):
        for j in range(size):
            text, image = gen_captcha(vocab[i], width, height)
            x[k] = np.array(image)
            y[k] = [i]
            k = k + 1
    return x, y


def save_np(gen_type, vocab, size, width, height, channel=3):
    x, y = get_np(vocab, size, width, height, channel)
    if gen_type == 0:
        np.save('data_np/x_train', x)
        np.save('data_np/y_train', y)
    elif gen_type == 1:
        np.save('data_np/x_test', x)
        np.save('data_np/y_test', y)
    else:
        np.save('data_np/x_test2', x)
        np.save('data_np/y_test2', y)


if __name__ == '__main__':
    # 0表示生成训练集，1表示生成测试集
    gen_type = 2
    vocab = '0123456789'
    size = 300
    width = 48
    height = 48
    save_np(gen_type, vocab, size, width, height)
