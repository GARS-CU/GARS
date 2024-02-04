import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
from keras import models
import sys
sys.path.insert(0, os.path.abspath("emotion"))
sys.path.insert(0, os.path.abspath("/.."))
from PAtt_Lite import Patt_Lite
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
sys.path.append(os.environ['GARS_PROJ'])
from util import *
path = os.path.join(var.GARS_PROJ, "datasets", "DAiSEE", "DataSet")


def get_open():
    scaler = MinMaxScaler()
    pca = PCA(n_components = 300)

    train_open = np.load(os.path.join(path, "Train", "openface_all_frames_features.npy"))

    #normalize the feature set
    train_open = scaler.fit_transform(np.reshape(train_open, (-1, 709)))

    #apply pca to get a 300x300 array of features
    train_open = pca.fit_transform(train_open)

    train_open = np.reshape(train_open, (-1, 300, 300, 1))

    val_open = np.load(os.path.join(path, "Validation", "openface_all_frames_features.npy"))
    val_open = scaler.transform(np.reshape(val_open, (-1, 709)))

    val_open = np.reshape(pca.transform(val_open), (-1, 300, 300, 1))

    return train_open, val_open

def assemble():

    train_emotion = np.load(os.path.join(path, "Train", "emotion_all_frames_features.npy"))

    val_emotion = np.load(os.path.join(path, "Validation", "emotion_all_frames_features.npy"))

    train_open, val_open = get_open()

    y_train = np.load(os.path.join(path, "Train", "Boredom_all_frames_Labels.npy"))
    y_val = np.load(os.path.join(path, "Validation", "Boredom_all_frames_Labels.npy"))

    return (train_emotion, train_open), y_train, (val_emotion, val_open), y_val

def build_dataset():
    x_train, y_train, x_val, y_val = assemble()

    y_train[y_train == 3] = 2
    y_val[y_val == 3] = 2

    balance_size = min([np.sum(y_train == categ) for categ in range(3)])
    indices = []
    for categ in range(3):
        ind = np.where(y_train == categ)[0]
        indices += np.random.choice(ind, size = balance_size, replace=  False).tolist()

    x_train = (x_train[0][indices], x_train[1][indices])
    
    y_train = y_train[indices]
    y_train = y_train/3 + (1/3)
    y_val = y_val/3  + (1/3)

    return x_train, y_train, x_val, y_val




#we need to define the custom attention class in order to load in the model weights for the emotion
#recognition model
class Custom_Attention(tf.keras.layers.Layer):

        def call(self, x):
            size = tf.shape(x)
            x = tf.reshape(x, [size[0], size[1], 1])
            y = tf.einsum("...ij,...jk->...ik", x, tf.transpose(x, perm = [0, 2, 1]))/16.0
            y = tf.matmul(tf.nn.softmax(y), x)
            return tf.reshape(y, [size[0], size[1]])

#we implement our focus and emotion classifiers as layers in our model
class Emotion_Regressor(Layer):
    def __init__(self, **kwargs):
        super(Emotion_Regressor, self).__init__(**kwargs)

        emoti_model = Patt_Lite().model
        tf.keras.utils.get_custom_objects()["Custom_Attention"] = Custom_Attention
        emoti_model.load_weights(os.path.join(var.GARS_PROJ, "Models", "Emotion_Rec", "PAtt_Lite_weights.h5"))

        inputs = keras.Input(shape = (48, 48, 1))
        
        x = emoti_model(inputs)
        
        #we attach a dense layer to our emotion classifier so that our emotion classifier can be trained
        #to detect engagement
        y = keras.layers.Dense(1)(x)

        self.model = Model(inputs, y)

    def __call__(self, x):
        return self.model(x)

#for our focus classifier, we implemented it as an MLP. Since we're already working with high level features,
#our classifier doesn't need to be as big as the emotion classifier
class Focus_Regressor(Layer):
    def __init__(self, **kwargs):
        super(Focus_Regressor, self).__init__(**kwargs)

        inputs = keras.layers.Input(shape = (300, 300, 1))
        x = keras.layers.Conv2D(32, 5, activation = "relu")(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, 5, activation = "relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(128, 5, activation = "relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(256, 5, activation = "relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation = "relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(1)(x)

        self.model = keras.models.Model(inputs,  x)

    def __call__(self, x):
        return self.model(x)


#Finally our aggregation layer goes through each of the 10 sampled / averaged frames
#and evaluates the focus and emotion classifiers on each of them
class AggregationLayer(Layer):

    def __init__(self, emo_model, open_model, num_frames, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.emoti_model = emo_model
        self.open_model = open_model
        self.num_frames = num_frames
    
    def __call__(self, inputs):

        #we evaluate our two classifiers on each frame
        emot_outputs = [self.emoti_model(frame) for frame in tf.unstack(inputs[0], axis=1)]
        focus_output = self.open_model(inputs[1])
        #[self.open_model(frame) for frame in tf.unstack(inputs[1], axis=1)]

        #average the emotion output
        aver_emot_output = tf.reduce_mean(tf.stack(emot_outputs), axis=0)

        #and then our output from the aggregation layer are the two 3x1 averaged outputs
        aggregate_output = tf.concat([aver_emot_output, focus_output], axis = 1)
        return aggregate_output#tf.math.reduce_mean(aggregate_output, axis = 1)


inp_emo = keras.Input((10, 48, 48, 1))
inp_open = keras.Input((300, 300, 1))

emoti_model = Emotion_Regressor()
focus_model = Focus_Regressor()

x = AggregationLayer(emoti_model, focus_model, 10)([inp_emo, inp_open])

output =  Dense(1)(x)

model = Model([inp_emo, inp_open], output)

model.summary()

model.compile(loss = "mean_squared_error",
              optimizer = keras.optimizers.AdamW(learning_rate = 1e-3),
              metrics = ["mse"])

x_train, y_train, x_val, y_val = build_dataset()

model.fit(x = x_train,
          y = y_train,
          validation_data = (x_val, y_val),
          epochs = 100,
          batch_size = 16
          )
