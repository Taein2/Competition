import os
import glob
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications import MobileNetV3Large

from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2


# tf.debugging.set_log_device_placement(True)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(42)
np.random.seed(42)
import time
import datetime

print(tf.__version__)
from tensorflow.python.client import device_lib


# 텐서를 GPU에 할당
with tf.device('/GPU:0'):
    device_lib.list_local_devices()

    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    BATCH_SIZE = 32
#    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#    base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
#    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.25))
#    model.add(BatchNormalization())
#    model.add(Dense(4, activation='tanh'))
#    model.add(Dropout(0.25))
#    model.add(BatchNormalization())
#    model.add(Dense(2, activation='softmax'))
#    model.add(Dropout(0.25))
#    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    train_path = 'F:/Competition_No1/datasets/train'
    test_path = 'F:/Competition_No1/datasets/test'

    train_gen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   brightness_range=[0.8, 1.2],
                                   validation_split=0.2)
    validation_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    test_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_gen.flow_from_directory(train_path,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    color_mode='rgb',
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    subset='training',
                                                    class_mode='binary',
                                                    )

    validation_generator = validation_gen.flow_from_directory(train_path,
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=False,
                                                              color_mode='rgb',
                                                              target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                              subset='validation',
                                                              class_mode='binary')

    zro = 0
    one = 0
    for i in range(len(train_generator.classes)):
        if train_generator.classes[i] == 0:
            zro += 1
        else:
            one += 1

    label = [0] * zro + [1] * one
    class_weight = compute_class_weight(class_weight = 'balanced', classes=np.unique(label), y=label)
    print(class_weight[0])
    print(class_weight[1])

    model.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    save_file_name = './ResNet152delrelu.h5'
    checkpoint = ModelCheckpoint(save_file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    earlystopping = EarlyStopping(monitor='val_loss', patience=5)

    start = time.time()
    hist = model.fit(train_generator, epochs=10, validation_data=validation_generator,
                     class_weight={0:class_weight[0],1:class_weight[1]},
                     callbacks=[checkpoint, earlystopping])
    end = time.time()

    sec = (end - start)
    r_time = datetime.timedelta(seconds=sec)
    print(r_time)