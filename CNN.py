from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from keras.utils import np_utils
from keras.optimizers import SGD

import numpy as np
import cv2
import time
import os
import random


class CNN():
    # 建構子：初始化
    def __init__(self, mode='test', img_source='camera'):
        self.img_shape = [28, 28]
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.mode = (mode == 'train')
        self.img_source = (img_source == 'camera')
        self.cap = None
        self.PATH = '/Users/chenyixuan/github/TEST/'
        self._ready()

    def _ready(self):
        self.build()
        if self.mode:
            self.load_train_data()
            self.train()
        else:
            if self.img_source:
                # 採用第0號攝影機
                self.cap = cv2.VideoCapture(0)
                # 設定影像的尺寸大小
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
                for i in range(100):
                    print('{}\t'.format(i + 1), end='')
                    self.predict(self.get_camera())
                    time.sleep(1)
                # 關閉攝影機
                self.cap.release()
            else:
                self.predict(self.get_folder())

    # 載入學習及測試數據
    def load_train_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.reshape(60000, self.img_shape[0], self.img_shape[1], 1)
        self.x_test = self.x_test.reshape(10000, self.img_shape[0], self.img_shape[1], 1)
        self.y_train = np_utils.to_categorical(self.y_train, 10)
        self.y_test = np_utils.to_categorical(self.y_test, 10)

    # 儲存影像
    def save_img(self, data):
        cv2.imwrite(self.PATH + 'save_imgs/{}.png'.format(time.strftime('%Y%m%d_%H%M%S',
                                                                        time.localtime()), random.randint(0, 1000)), data)

    # 擷取攝影機畫面
    def get_camera(self):
        # 擷取畫面
        frame = self.cap.read()[1]
        # 壓縮影像
        frame = cv2.resize(frame, (self.img_shape[0], self.img_shape[1]))
        frame = self.reshape(frame)
        return frame

    # 載入`/test_imgs`所有影像
    def get_folder(self):
        files = os.listdir(self.PATH + 'test_imgs')
        imgs = np.empty(shape=[len(files), self.img_shape[0], self.img_shape[1], 1])
        for i, file in enumerate(files):
            imgs[i] = self.reshape(cv2.imread(self.PATH + 'test_imgs/{}'.format(file)), single=False)
        return imgs

    # 轉換大小
    def reshape(self, img, single=True):
        temp = np.empty(shape=[self.img_shape[0], self.img_shape[1], 1])
        for i, c in enumerate(img):
            for j, r in enumerate(c):
                v = (int(r[0]) + int(r[1]) + int(r[2])) / 3.0
                temp[i, j, 0] = 255 - v
        if single:
            return np.array([temp])
        else:
            return temp

    # 建構模型
    def build(self):
        if os.path.isfile(self.PATH + 'model/model_cnn.json') and os.path.isfile(self.PATH + 'model/weights_cnn.h5'):
            print('>>> Loading weight ...')
            self.model = model_from_json(open(self.PATH + 'model/model_cnn.json').read())
            self.model.load_weights(self.PATH + 'model/weights_cnn.h5')
        else:
            model = Sequential()
            self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(
                self.img_shape[0], self.img_shape[1], 1)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, (3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(128, (3, 3), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(200))
            self.model.add(Activation('relu'))
            self.model.add(Dense(10))
            self.model.add(Activation('softmax'))

    # 學習
    def train(self):
        self.model.compile(loss='mse', optimizer=SGD(lr=0.05), metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, batch_size=100, epochs=12)
        model_json = self.model.to_json()
        open(self.PATH + 'model/model_cnn.json', 'w').write(model_json)
        self.model.save_weights(self.PATH + 'model/weights_cnn.h5')

    # 預測
    def predict(self, img):
        print(self.model.predict_classes(img))


cnn = CNN(mode='test', img_source='camera')
