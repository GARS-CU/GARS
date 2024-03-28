import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
# import sys
# sys.path.append(os.environ['GARS_PROJ'])
# sys.path.insert(0, os.path.abspath("../.."))
# from util import *

class Attention(tf.keras.layers.Layer):

        def call(self, x):
            size = tf.shape(x)
            x = tf.reshape(x, [size[0], size[1], 1])
            y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
            y = tf.matmul(tf.nn.softmax(y), x)
            return tf.reshape(y, [size[0], size[1]])


class Patt_Lite:
     

    def __init__(self):
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
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.DepthwiseConv2D( (10,10), strides = 2)(x)
        x = tf.keras.layers.Conv2D(256, (1,1), activation = "relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.DepthwiseConv2D((3,3))(x)
        x = tf.keras.layers.Conv2D(16, (1,1), activation = "relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation = "relu")(x)
        x = Attention()(x)
        outputs = tf.keras.layers.Dense(10, activation = "softmax")(x)

        self.model = tf.keras.Model(inputs, outputs)

        


    # def train(self):
    #     x_train = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xtrain.npy"))
    #     y_train = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_ytrain.npy"))
    #     x_test = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xtest.npy"))
    #     y_test = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_ytest.npy"))    


    #     y_train = np.argmax(y_train, axis = 1)
    #     y_test = np.argmax(y_test, axis = 1)

    #     print(y_test)
    #     print(y_test.shape)

    #     opt = tf.keras.optimizers.AdamW(learning_rate = 1e-3)

    #     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


    #     self.model.compile(optimizer = opt, 
    #             loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    #             #loss = tf.keras.losses.BinaryCrossentropy(), 
    #             metrics = ["accuracy"])

    #     history = self.model.fit(x_train, y_train, epochs = 125, batch_size = 10, 
    #         validation_data = (x_test, y_test), callbacks = [callback])
        
    #     plt.plot(history.history["accuracy"])
    #     plt.plot(history.history["val_accuracy"])
    #     plt.title("Model Accuracy")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Accuracy")
    #     plt.legend(["Train", "Validation"], loc = "upper left")

    #     plt.savefig(os.path.join(var.GARS_PROJ, "Models", "Emotion_Rec", "PAtt_Lite_Accuracy.pdf"))
        
    #     self.model.save(os.path.join(var.GARS_PROJ, "Models", "Emotion_Rec", "PAtt_Lite_weights.h5"))
        
