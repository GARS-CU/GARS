import tensorflow as tf
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append(os.environ['GARS_PROJ'])
from util import *

class Custom_Attention(tf.keras.layers.Layer):

        def call(self, x):
            size = tf.shape(x)
            x = tf.reshape(x, [size[0], size[1], 1])
            y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
            y = tf.matmul(tf.nn.softmax(y), x)
            return tf.reshape(y, [size[0], size[1]])

class Patt_Lite:
    def __init__(self, dropout_rate=0.2):
        mobile_net = tf.keras.applications.mobilenet.MobileNet(
        include_top=False)
        
        mobile_net.trainable = False
        inputs = tf.keras.Input(shape=(48, 48, 1), name="image")
        x = tf.keras.layers.Conv2D(3, (1, 1))(inputs)
        x = tf.keras.layers.Resizing(224, 224)(x)
        x = tf.keras.applications.mobilenet.preprocess_input(x)

        for layer in mobile_net.layers[:56]:
            x = layer(x)

        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.DepthwiseConv2D((10, 10), strides=2)(x)
        x = tf.keras.layers.Conv2D(256, (1, 1), activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.DepthwiseConv2D((3, 3))(x)
        x = tf.keras.layers.Conv2D(16, (1, 1), activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = Custom_Attention()(x)
        outputs = tf.keras.layers.Dense(9, activation="softmax")(x)

        self.model = tf.keras.Model(inputs, outputs)

    def train(self, learning_rate=1e-3):
        x_train = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xtrain.npy"))
        y_train = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_ytrain.npy"))
        x_test = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xtest.npy"))
        y_test = np.load(os.path.join(var.GARS_PROJ, "datasets", "FERP", "FERP_xtest.npy"))    

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.model.compile(optimizer=opt, 
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                           metrics=["accuracy"])

        history = self.model.fit(x_train, y_train, epochs=100, batch_size=64, 
            validation_data=(x_test, y_test), callbacks=[callback])

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"], loc="upper left")
        
        plt.savefig(os.path.join(var.GARS_PROJ, "Models", "Emotion_Rec", "PAtt_Lite_Accuracy_Pres.pdf"))
        
        tf.keras.utils.get_custom_objects()["Custom_Attention"] = Custom_Attention
       
        self.model.save_weights(os.path.join(var.GARS_PROJ, "Models", "Emotion_Rec", "PAtt_Lite_weights_safety.h5"))
        return history

def grid_search(dropout_rates, learning_rates):
    results = []
    for dropout_rate in dropout_rates:
        for lr in learning_rates:
            print(f"Training with dropout_rate={dropout_rate}, learning_rate={lr}")
            model = Patt_Lite(dropout_rate=dropout_rate)
            history = model.train(learning_rate=lr)

            results.append({
                "dropout_rate": dropout_rate,
                "learning_rate": lr,
                "val_accuracy": max(history.history['val_accuracy']),
                "val_loss": min(history.history['val_loss'])
            })

    return pd.DataFrame(results)

# Regular run of model

# model = Patt_Lite(dropout_rate=0.2)
# model.train(learning_rate=0.1)

# Grid search

dropout_rates = [0.2]
learning_rates = [1e-3]
df = grid_search(dropout_rates, learning_rates)
csv_path = os.path.join(var.GARS_PROJ, "results", "pattlite_param", "gridsearch_lr1e-4.csv")

df.to_csv(csv_path, index=False)
