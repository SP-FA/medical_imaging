import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

class InputImg:
    data = []
    def __init__(self, data_dir, width): #文件地址 './Data'
        classes = os.listdir(data_dir)
        k = 0
        for name in classes:
            name_ = os.listdir(data_dir+'/'+name)
            for two in name_:
                two_ = os.listdir(data_dir+'/'+name+'/'+two)
                for id in two_:
                    pictures = os.listdir(data_dir+'/'+name+'/'+two+'/'+id)
                    for pic in pictures:
                        img = Image.open(data_dir+'/'+name+'/'+two+'/'+id+'/'+pic)
                        img = np.array(img.resize((width, width), Image.ANTIALIAS))
                        if len(img.shape) == 2:
                            continue;
                        c = self.turn_img(img, width, 1)
                        self.data.append([img, k])
                        self.data.append([c  , k])
            k += 1

    def load_data(self, size = 0.1):
        random.shuffle(self.data)
        num = round(len(self.data) * size)
        x = []
        y = []
        for i in self.data:
            x.append(i[0])
            y.append(i[1])
        y = self.one_hot(y)
        train_x = tf.convert_to_tensor(x[num:])
        train_y = tf.convert_to_tensor(y[num:])
        test_x  = tf.convert_to_tensor(x[:num])
        test_y  = tf.convert_to_tensor(y[:num])
        return train_x, train_y, test_x, test_y

    def turn_img(self, img, width, case):
        c = Image.new("RGB", (width, width))
        for i in range(width):
            for j in range(width):
                w = width - i - 1
                h = width - j - 1
                if   case == 1:
                    rgb = tuple(img[j, w]) #镜像翻转 
                elif case == 2:
                    rgb = tuple(img[h, w]) #翻转180度  
                elif case == 3:
                    rgb = tuple(img[h, i]) #上下翻转  
                c.putpixel([i, j], rgb)
        c = np.array(c)
        return c

    def preprocess(self, x, y):
        x = 2 * tf.cast(x, dtype = tf.float32) / 255.-1
        y = tf.cast(y, dtype = tf.int32)
        return x, y

    def one_hot(self, y):
        _y = []
        for i in y:
            if   i == 0:
                _y.append([1, 1]) #CHC
            elif i == 1:
                _y.append([0, 1]) #HCC
            elif i == 2:
                _y.append([1, 0]) #ICC
        return _y

