import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from pprint import pprint
import sys
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import keras
import os
import numpy as np
import tensorflow as tf
from keras.optimizers import *
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if len(sys.argv) == 1:
    print("Needs to be called with a parameter")
    sys.exit(10)

height = 64
width = 64

model = None
if os.path.isdir('savedmodel'):
    model = keras.models.load_model('savedmodel')

    img_path = sys.argv[1]
    img = keras.preprocessing.image.img_to_array(image.load_img(img_path, target_size=(width, height)))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)

    empty = pred[0][0]

    if empty >= 0.8:
        print("Empty")
        sys.exit(0)
    else:
        print("Nonempty")
        sys.exit(1)
else:
    epochs = 15
    image_data_generator = keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    image_set = image_data_generator.flow_from_directory('images/', target_size=(width, height), color_mode='rgb')
    model = keras.models.Sequential()
    model.add(Conv2D(input_shape=[width, height, 3],
        trainable=True,
        use_bias=True,
        activation="sigmoid",
        padding="valid",
        filters=16,
        kernel_size=[2,2],
        strides=[2,2],
        dilation_rate=[1,1],
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling",
        dtype="float32"
    ))
    model.add(Conv2D(trainable=True,
        use_bias=True,
        activation="linear",
        padding="valid",
        filters=8,
        kernel_size=[2,2],
        strides=[2,2],
        dilation_rate=[1,1],
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling"
    ))
    model.add(Flatten())
    model.add(Dense(trainable=True,
        use_bias=True,
        units=64,
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling"
    ))
    model.add(Dense(trainable=True,
        use_bias=True,
        units=2,
        activation="softmax",
        kernel_initializer="glorot_uniform",
        bias_initializer="variance_scaling"
    ))
    opt = 'adam'
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
    model.summary()
    model.fit_generator(image_set, epochs=epochs)
    model.save('savedmodel')
