from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.python.client import device_lib
import time
import numpy as np
import pandas as pd
tf.random.set_seed(42)
np.random.seed(42)

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# 텐서를 GPU에 할당
with tf.device('/GPU:0'):
    device_lib.list_local_devices()
    tf.random.set_seed(42)

    test_path = 'F:/Competition_No1/datasets/test'
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_directory(test_path,
                                                    batch_size=BATCH_SIZE,
                                                    color_mode='rgb',
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    shuffle=False,
                                                    class_mode='binary'
                                                    )
    model = load_model('ResNet152delrelu.h5')

    start = time.time()
    y_pred = model.predict_classes(test_generator)
    end = time.time()

    image_list = []
    gt_val = []
    prediction_val = []
    accuracy_val = []
    for i in range(len(y_pred)):
        if test_generator.classes[i] == y_pred[i]:
            accuracy_val.append("True")
        else:
            accuracy_val.append("False")
        image_list.append(test_generator.filenames[i])
        gt_val.append(test_generator.classes[i])
        prediction_val.append(y_pred[i])

    df = pd.DataFrame(zip(image_list, gt_val, prediction_val, accuracy_val),
                       columns=['Image list', 'GT', 'Prediction', 'accuracy'])
    print(df)
    inference_time = (end-start) / (i+1)
    df=df.append({"Image list": "Average Accuracy", "GT": accuracy_val.count("True")/len(accuracy_val)},ignore_index=True)
    df=df.append({"Image list": "Inference Speed(ms)", "GT": inference_time * 1000},ignore_index=True)
    print("time/image: ", inference_time)
    df.to_csv("../datasets/test/result.csv")
