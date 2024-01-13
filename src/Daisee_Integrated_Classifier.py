import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
from keras import models
from PAtt_Lite import Patt_Lite

#This program implements and trains the complete engagement detection model (emotion + focus classifier) on the DAiSEE dataset



def build_dataset(dataset):
    path = os.path.join("..", "..", "datasets", "DAiSEE", "DataSet", dataset)

    open_features = np.load(os.path.join(path, "open_features.npy"))
    emotion_features = np.load(os.path.join(path, "emotion_features.npy"))
    labels = np.load(os.path.join(path, "labels.npy"))

    return [emotion_features, open_features], labels


x_train, y_train = build_dataset("Train")
x_val, y_val = build_dataset("Validation")

print(len(x_train))
print(x_train[0].shape)


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
class Emotion_Classifier(Layer):
    def __init__(self, **kwargs):
        super(Emotion_Classifier, self).__init__(**kwargs)

        emoti_model = Patt_Lite().model
        tf.keras.utils.get_custom_objects()["Custom_Attention"] = Custom_Attention
        emoti_model.load_weights("../../Models/Emotion_Rec/PAtt_Lite_weights.h5")

        inputs = keras.Input(shape = (48, 48, 1))
        
        y = emoti_model(inputs)
        
        #we attach a dense layer to our emotion classifier so that our emotion classifier can be trained
        #to detect engagement
        y = keras.layers.Dense(3, activation = "relu")(y)

        self.model = Model(inputs, y)

    def __call__(self, x):
        return self.model(x)

#for our focus classifier, we implemented it as an MLP. Since we're already working with high level features,
#our classifier doesn't need to be as big as the emotion classifier
class Focus_Classifier(Layer):
    def __init__(self, **kwargs):
        super(Focus_Classifier, self).__init__(**kwargs)
        inputs = keras.Input(shape = (329))
        y = keras.layers.Dense(64, activation = "relu")(inputs)
        y = keras.layers.Dense(64, activation = "relu")(y)
        y = keras.layers.Dense(128, activation = "relu")(y)
        y = keras.layers.Dense(3, activation = "relu")(y)
        self.model = Model(inputs, y)

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
        focus_outputs = [self.open_model(frame) for frame in tf.unstack(inputs[1], axis=1)]

        #average the outputs
        aver_emot_output = tf.reduce_mean(tf.stack(emot_outputs), axis=0)
        aver_focus_output = tf.reduce_mean(tf.stack(focus_outputs), axis=0)

        #and then our output from the aggregation layer are the two 3x1 averaged outputs
        aggregate_output = tf.concat([aver_emot_output, aver_focus_output], axis = 1)
        return aggregate_output

#our model takes in two inputs, the cropped image for the emotion classifier and the 
#10x35 array of features for our focus classifier
    
inp_emo = keras.Input((10, 48, 48, 1))
inp_open = keras.Input((10, 329))


emoti_model = Emotion_Classifier()
focus_model = Focus_Classifier()


y = AggregationLayer(emoti_model, focus_model, 10)([inp_emo, inp_open])

#we add a dense layer after our aggregation model so that the outputs from the emotion and focus classifiers
#are combined together
outputs = Dense(3, activation = "softmax")(y)


model = Model([inp_emo, inp_open], outputs)

model.summary()

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 1e-2),
              metrics = ["acc"])

model.fit(x = x_train,
          y = y_train,
          validation_data = (x_val, y_val),
          epochs = 50,
          batch_size = 32
          )
