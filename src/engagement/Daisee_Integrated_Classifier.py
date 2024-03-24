import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
from keras import models
import sys
sys.path.insert(0, os.path.abspath("emotion"))
sys.path.insert(0, os.path.abspath("../"))
from PAtt_Lite import Patt_Lite
import pandas as pd
sys.path.append(os.environ['GARS_PROJ'])
from util import *

#This program implements and trains the complete engagement detection model (emotion + focus classifier) on the DAiSEE dataset



def build_dataset(dataset):
    path = os.path.join(var.GARS_PROJ, "datasets", "DAiSEE", "DataSet", dataset)
    
    
    open_features = np.load(os.path.join(path, "open_allfeatures.npy"))
    open_features = np.reshape(tf.keras.utils.normalize(np.reshape(open_features, (-1, 709))), (-1, 10, 709)) 
    
    emotion_features = np.load(os.path.join(path, "emotion_allfeatures.npy"))
    
    labels = np.load(os.path.join(path, "labels.npy"))
    #df = pd.read_csv(os.path.join("..", "..", "datasets", "DAiSEE", "Labels", dataset + "Labels.csv"))
    #labels = df["Boredom"].to_numpy()
    
    labels[labels == 3] = 2
    #labels[labels == 1] = 0
    #labels[labels == 2] = 1
    #labels[labels == 3] = 2

    #if dataset == "Train":
    
    print(open_features.shape)
    print(labels.shape)
    balance_size = min([np.sum(labels == categ) for categ in range(3)])
    print(balance_size)
    indices = []
    for categ in range(3):
        ind = np.where(labels == categ)[0]
        indices += np.random.choice(ind, size = balance_size, replace = False).tolist()
    open_features = open_features[indices]
    emotion_features = emotion_features[indices]
    labels = labels[indices]
    
    #labels[labels == 3] = 2
    
    emotion_features = emotion_features.reshape(-1, 48, 48)
    
    open_features = open_features.reshape(-1, 709)
    

    labels = np.array(sum([[labels[index] for i in range(10)] for index in range(len(labels))], []))

    labels = labels/3 + (1/3)
    
    return [emotion_features, open_features], labels


x_train, y_train = build_dataset("Train")
print(x_train[0].shape)
x_val, y_val = build_dataset("Validation")


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
        emoti_model.load_weights(f"{var.GARS_PROJ}/Models/Emotion_Rec/PAtt_Lite_weights.h5")
        
        emoti_model.summary()
        #for index in range(len(emoti_model.layers)):
            #emoti_model.layers[index].trainable = False
        #print(emoti_model.layers[:-1])
        
        #emoti_model = tf.keras.models.Sequential(emoti_model.layers[:-1])
        #emoti_model.summary()
        inputs = keras.Input(shape = (48, 48, 1))
        x = emoti_model.layers[1](inputs)
        x = tf.keras.layers.Resizing(224, 224)(x)
        x = tf.keras.applications.mobilenet.preprocess_input(x)
        
        for layer in emoti_model.layers[5:-1]:
            print(layer.name)
            x = layer(x)

        #x = tf.keras.applications.mobilenet.preprocess_input(x)
        #for layer in emoti_model.layers[5:-1]:
            #x = layer(x)
        
        #x = emoti_model(inputs)
        
        #we attach a dense layer to our emotion classifier so that our emotion classifier can be trained
        #to detect engagement
        #y = keras.layers.Dense(256, activation = "relu")(x)
        #x = keras.layers.Dense(256, activation = "relu")(x)
        #x = keras.layers.Dense(256, activation = "relu")(x)
        #y = keras.layers.Dense(256, activation = "relu")(x)
        self.model = Model(inputs, x)
        #self.model.summary()

    def __call__(self, x):
        return self.model(x)

#for our focus classifier, we implemented it as an MLP. Since we're already working with high level features,
#our classifier doesn't need to be as big as the emotion classifier
class Focus_Classifier(Layer):
    def __init__(self, **kwargs):
        super(Focus_Classifier, self).__init__(**kwargs)
        inputs = keras.Input(shape = (709))
        y = keras.layers.Dense(256, activation = "relu")(inputs)
        y = keras.layers.Dense(256, activation = "relu")(y)

        #y = keras.layers.Dense(1)(y)
        #y = keras.layers.Dense(256, activation = "relu")(y)
        #y = keras.layers.Dense(3, activation = "relu")(y)
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
        #emot_outputs = [self.emoti_model(frame) for frame in tf.unstack(inputs[0], axis=1)]
        #focus_outputs = [self.open_model(frame) for frame in tf.unstack(inputs[1], axis=1)]
        
        emot_output = self.emoti_model(inputs[0])
        focus_output = self.open_model(inputs[1])
        #average the outputs
        #aver_emot_output = tf.reduce_mean(tf.stack(emot_outputs), axis=0)
        #aver_focus_output = tf.reduce_mean(tf.stack(focus_outputs), axis=0)

        #and then our output from the aggregation layer are the two 3x1 averaged outputs
        aggregate_output = tf.concat([emot_output, focus_output], axis = 1)
        return aggregate_output#aver_emot_output#aggregate_output

#our model takes in two inputs, the cropped image for the emotion classifier and the 
#10x35 array of features for our focus classifier
    
inp_emo = keras.Input((48, 48, 1))
inp_open = keras.Input((709))


emoti_model = Emotion_Classifier()
focus_model = Focus_Classifier()

y = AggregationLayer(emoti_model, focus_model, 10)([inp_emo, inp_open])

#we add a dense layer after our aggregation model so that the outputs from the emotion and focus classifiers
#are combined together
y = Dense(128, activation = "relu")(y)
outputs = Dense(1)(y)


model = Model([inp_emo, inp_open], outputs)

model.summary()

model.compile(loss = "mean_squared_error",
        
        optimizer = keras.optimizers.AdamW(learning_rate = 1e-3),
              metrics = ["mse"])

#print(y_val)
#results = model.predict(x_val)
#results = np.isnan(np.sum(results))

#for res in results:
#    print(res)
"""
scc= tf.keras.losses.SparseCategoricalCrossentropy()

for i in range(len(y_val)):
    if np.isnan(scc(y_val[i], results[i])):
        print(results[i], y_val[i])

"""
#print(tf.keras.losses.SparseCategoricalCrossentropy()(y_val[0], results[0]))

model.fit(x = x_train,
          y = y_train,
          validation_data = (x_val, y_val),
          epochs = 1600,
          batch_size = 16
          )
model.save("integrated_model.h5")
