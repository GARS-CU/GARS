import numpy as np
import pickle
import os
import sys
sys.path.insert(0, os.path.abspath("../"))
sys.path.append(os.environ['GARS_PROJ'])
from util import *
from sklearn.utils import shuffle
import keras
import keras_nlp
import tensorflow.keras.backend as k
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.utils import Sequence

####TensorFlow GPU Configuration#######

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
##########################################################

#this sets the sequence length for the input to our model
#with a sequence length of 10 i think there were around 90,000 sequences
#which should be large enough to train on i think but if not we can try 
#decreasing the sequence length
SEQUENCE_LENGTH = 10
PADDING = np.reshape(np.array([0 for i in range(4096)]), (1, 4096))
START = np.reshape(np.array([-1 for i in range(4096)]), (1, 4096))
#path to the behance dataset files
rec_path = os.path.join(var.GARS_PROJ, "datasets", "Behance")

INPUT_DIM = 700
BATCH_SIZE  =10
def groupAppreciates():
        #we load in the dictionary which maps each user id to the list of indices into the numpy array
    #that contains the feature vector for each item
    with open(os.path.join(rec_path, "Behance_user_item_appreciates.pkl"), "rb") as f:
        user_appreciates = pickle.load(f)

    users = list(user_appreciates.keys())

    #for each user
    for user in users:
        #if they didn't interact with <= half of our sequence length then
        if len(user_appreciates[user]) <= SEQUENCE_LENGTH // 2:
            #then we remove them from the dictionary since if we added them more than half of the input to the transformer
            #would just be padding
            del user_appreciates[user]
        #if the list is slightly shorter than sequence length
        elif len(user_appreciates[user]) <= SEQUENCE_LENGTH:
            #we pad the end with 0s
            user_appreciates[user] += [0 for i in range(SEQUENCE_LENGTH - len(user_appreciates[user]))]
        else:
            #else we need to break up the user's interactions into chunks of length sequence length

        
            for index in range(0, len(user_appreciates[user]), SEQUENCE_LENGTH):
                #we take each chunk and map it to a new user entry
                user_appreciates[user + "v" + str(index)] = user_appreciates[user][index:index+SEQUENCE_LENGTH]
        
            #if the last chunk is large enough
            if len(user_appreciates[user]) % SEQUENCE_LENGTH > SEQUENCE_LENGTH // 2:
                #we pad it with zeros
                user_appreciates[user + "v" + str(index)] += [0 for i in range(SEQUENCE_LENGTH - len(user_appreciates[user]) % SEQUENCE_LENGTH)]
            else:
                #else we delete it
                del user_appreciates[user + "v" + str(index)]
                                                        
            #and then delete the original user entry
            del user_appreciates[user]
    
    for index, user in enumerate(user_appreciates):
        if len(user_appreciates[user]) != SEQUENCE_LENGTH:
            print("ERROR")
            break
        #[START, 1, 2, 3, 4]
        #[1, 2, 3, 4, 5]
        source = user_appreciates[user][1:]
        source.insert(0, -2)
        target = user_appreciates[user]
    
        user_appreciates[user] = (source, target)

    return user_appreciates

def decompose(x, y):
    x, y = shuffle(x, y)
    x_train = x[:int(.8*len(x))]
    y_train = y[:int(.8*len(y))]

    x_val = x[int(.8*len(x)):int(.9*len(x))]
    y_val = y[int(.8*len(x)):int(.9*len(y))]

    x_test = x[int(.9*len(x)):]
    y_test = y[int(.9*len(y)):]
    
    scaler = StandardScaler()
    scaler.fit(np.reshape(x_train, (-1, 4096)))
    pca = PCA(n_components = INPUT_DIM)

    x_train = np.reshape(pca.fit_transform(scaler.transform(np.reshape(x_train, (-1, 4096)))), (-1, SEQUENCE_LENGTH, INPUT_DIM))
    x_val = np.reshape(pca.transform(scaler.transform(np.reshape(x_val, (-1, 4096)))), (-1, SEQUENCE_LENGTH, INPUT_DIM))
    x_test = np.reshape(pca.transform(scaler.transform(np.reshape(x_test, (-1, 4096)))), (-1, SEQUENCE_LENGTH, INPUT_DIM))

    y_train = np.reshape(pca.fit_transform(scaler.transform(np.reshape(y_train, (-1, 4096)))), (-1, SEQUENCE_LENGTH, INPUT_DIM))
    y_val = np.reshape(pca.fit_transform(scaler.transform(np.reshape(y_val, (-1, 4096)))), (-1, SEQUENCE_LENGTH, INPUT_DIM))
    y_test = np.reshape(pca.fit_transform(scaler.transform(np.reshape(y_test, (-1, 4096)))), (-1, SEQUENCE_LENGTH, INPUT_DIM))

    return x_train, y_train, x_val, y_val, x_test, y_test

    

def prepare_data():

    user_appreciates = groupAppreciates()
    item_features = np.load(os.path.join(rec_path, "behance_item_features.npy"))
    
    #item_features = decompose(item_features)
    item_features = np.concatenate((item_features, START))
    item_features = np.concatenate((item_features, PADDING))
    
    x = np.zeros((len(user_appreciates), SEQUENCE_LENGTH, 4096))
    y = np.zeros((len(user_appreciates), SEQUENCE_LENGTH, 4096))

    for index, user in enumerate(user_appreciates):

        x[index] = item_features[user_appreciates[user][0]]
        y[index] = item_features[user_appreciates[user][1]]
    
    return decompose(x,y)

x_train, y_train, x_val, y_val, x_test, y_test=  prepare_data()

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# Create a simple model containing the decoder.

decoder = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=256, num_heads=8)

decoder_input = keras.Input(shape=(SEQUENCE_LENGTH, INPUT_DIM))
#encoder_input = keras.Input(shape=(SEQUENCE_LENGTH, INPUT_DIM))
x = decoder(decoder_input)
#output = keras.layers.Dense(INPUT_DIM, activation = "relu")(x)

model = keras.Model(
    inputs = decoder_input,
    outputs=x
)

train_generator = DataGenerator(x_train, y_train, BATCH_SIZE)
val_generator = DataGenerator(x_val, y_val, BATCH_SIZE)

#decoder_output = model((decoder_input_data, encoder_input_data))

model.summary()
model.compile(
    loss="mse",
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
)

model.fit_generator(
    train_generator,
    validation_data = val_generator,
    epochs=150
)


