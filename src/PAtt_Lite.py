import tensorflow as tf
import os
import numpy as np
from PIL import Image


classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
path_dataset = os.path.abspath("../datasets/FER2013")

def load_data(subset):
    pathname = os.path.join(path_dataset, subset)
    data = np.zeros((1, 48, 48))
    labels = []

    for index in range(len(classes)):
        class_path = os.path.join(pathname, classes[index])
        for path in os.listdir(class_path):

            img = Image.open(os.path.join(class_path, path))
            data = np.concatenate((data, np.asarray(img).reshape(1, 48, 48)))
            labels.append(index)
    
    return data[1:], np.asarray(labels)

x_train, y_train = load_data("train")

x_test, y_test = load_data("test")

mobile_net = tf.keras.applications.mobilenet.MobileNet(
    include_top = False)


mobile_net.trainable = False
inputs = tf.keras.Input(shape = (48, 48, 1), name = "image")
x = tf.keras.layers.Conv2D(3, (1, 1))(inputs)
x = tf.keras.layers.Resizing(224, 224)(x)
x = tf.keras.applications.mobilenet.preprocess_input(x)

count = 0
for layer in mobile_net.layers[:56]:
    x = layer(x)

x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)

x = tf.keras.layers.DepthwiseConv2D( (10,10), strides = 2)(x)
x = tf.keras.layers.Conv2D(256, (1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3))(x)
x = tf.keras.layers.Conv2D(256, (1,1))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(16, activation = "relu")(x)
x = tf.keras.layers.Attention(use_scale = True)([x, x])
x = tf.keras.layers.Dense(16, activation = "relu")(x)
outputs = tf.keras.layers.Dense(7, activation = "softmax")(x)
model = tf.keras.Model(inputs, outputs)

print(model.summary())

model.compile(optimizer = "adam", 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 10,
          batch_size = 128, validation_data = (x_test, y_test))
