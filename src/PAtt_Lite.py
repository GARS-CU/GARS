import tensorflow as tf
import os
import numpy as np


class Attention(tf.keras.layers.Layer):

    def call(self, x):
        size = tf.shape(x)
        x = tf.reshape(x, [size[0], size[1], 1])
        y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
        y = tf.matmul(y, x)
        return tf.reshape(y, [size[0], size[1]])

        
        
        

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")


mobile_net = tf.keras.applications.mobilenet.MobileNet(
    include_top = False)


mobile_net.trainable = False
inputs = tf.keras.Input(shape = (48, 48, 1), name = "image")
x = tf.keras.layers.Conv2D(3, (1, 1))(inputs)
x = tf.keras.layers.Resizing(224, 224)(x)
x = tf.keras.applications.mobilenet.preprocess_input(x)

for layer in mobile_net.layers[:56]:
    x = layer(x)

    

x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.DepthwiseConv2D( (10,10), strides = 2)(x)
x = tf.keras.layers.Conv2D(256, (1,1), activation = "relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.DepthwiseConv2D((3,3))(x)
x = tf.keras.layers.Conv2D(16, (1,1), activation = "relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation = "relu")(x)
#x = tf.keras.layers.Attention(use_scale = True)([x, x])
x = Attention()(x)
#x = tf.keras.layers.Dense(256, activation = "relu")(x)
outputs = tf.keras.layers.Dense(7, activation = "softmax")(x)
model = tf.keras.Model(inputs, outputs)

print(model.summary())

opt = tf.keras.optimizers.AdamW()

model.compile(optimizer = opt, 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 30,
          batch_size = 64, validation_data = (x_test, y_test))

print(model.predict(x_test))