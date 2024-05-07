import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
from keras import models
import sys
from PAtt_Lite import Patt_Lite
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as pk

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
        emoti_model.load_weights(os.path.join("Models", "PAtt_Lite_weights.h5"))

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
        x = keras.layers.Conv2D(128, 5, activation = "relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation = "relu")(x)
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


class Engagement_Classifier:

    def __init__(self):

        inp_emo = keras.Input((10, 48, 48, 1))
        inp_open = keras.Input((300, 300, 1))

        emoti_model = Emotion_Regressor()
        focus_model = Focus_Regressor()

        x = AggregationLayer(emoti_model, focus_model, 10)([inp_emo, inp_open])

        output =  Dense(1)(x)

        self.model = Model([inp_emo, inp_open], output)

# Class just for inference - loads in weights and uses model to predict
class EngagementClassifierInference:
    def __init__(self):
        # Ensure that the model initialization also happens on the CPU
        with tf.device('/cpu:0'):
            inp_emo = Input((10, 48, 48, 1))
            inp_open = Input((300, 300, 1))

            emoti_model = Emotion_Regressor()
            focus_model = Focus_Regressor()

            x = AggregationLayer(emoti_model, focus_model, 10)([inp_emo, inp_open])

            output = Dense(1)(x)

            self.model = Model([inp_emo, inp_open], output)
            
            weights_path = os.path.join("Models", "Engagement_Model_weights.h5")
            self.model.load_weights(weights_path)

    def predict_engagement(self, emotion_features, openface_features):
        with tf.device('/cpu:0'):  # Explicitly using CPU for inference
            emotion_features = np.expand_dims(emotion_features, axis=0)
            openface_features = np.expand_dims(openface_features, axis=0)
            
            engagement_score = self.model.predict([emotion_features, openface_features])
            return engagement_score

    def get_open_inference(self, open_features):
        with tf.device('/cpu:0'):  # Also ensure this part runs on CPU if it involves any computation
            # Get saved scaler and pca file paths
            pca_path = os.path.join("Models", "pca.pkl")

            # Load the PCA models
            scaler_path = os.path.join("Models", "scaler.pkl")
            with open(pca_path, 'rb') as f:
                pca = pk.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pk.load(f)
            #scaler = MinMaxScaler()
            #pca = PCA(n_components = 300)
            # Normalize the feature set and apply pca
            open_features_normalized = scaler.transform(open_features)
            open_features_pca = pca.transform(open_features_normalized)

            return open_features_pca


# x_train, y_train, x_val, y_val = build_dataset()

# minLoss = float("inf")
# for step_size in np.arange(1e-4, 1e-3, 1e-4):
#         classifier = Engagement_Classifier();
#         classifier.model.summary();
#         opt = tf.keras.optimizers.AdamW(learning_rate = step_size)

#         callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
#         classifier.model.compile(optimizer = opt,
#             loss = "mse",
#             #loss = tf.keras.losses.BinaryCrossentropy(),
#             metrics = ["mse"])

#         history = classifier.model.fit(x_train, y_train, epochs = 100, batch_size = 16,
#                 validation_data = (x_val, y_val), callbacks = [callback])
        
#         loss = classifier.model.evaluate(x_val, y_val)[0]

#         print(loss)

#         if loss < minLoss:
#             minLoss = loss
#             optSize = step_size
#             optHistory = history
#             optModel = classifier.model



# plt.plot(history.history["mse"])
# plt.plot(history.history["val_mse"])
# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(["Train", "Validation"], loc = "upper left")

# plt.savefig(os.path.join(var.GARS_PROJ, "Models", "Integrated_Model", "Engagement_Model_Results.pdf"))

# optModel.save(os.path.join(var.GARS_PROJ, "Models", "Integrated_Model", "Engagement_Model_weights.h5"))
