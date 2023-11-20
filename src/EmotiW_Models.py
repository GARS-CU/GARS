import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from sklearn.decomposition import PCA


#Here I tested out a couple of models on the dataset (but not really thoroughly)

#load in the datasets
x_train = np.load("EmotiW_x_train.npy")
y_train  = np.load("EmotiW_y_train.npy")
x_test = np.load("EmotiW_x_test.npy")
y_test = np.load("EmotiW_y_test.npy")

#this sets the number of features used by the models. A lot of the open face features seemed somewhat
#redundant or highly correlated with one another so I thought that doing PCA to reduce the number of features
#could improve the accuracy of the ML models (I don't know if it would help the DL models that much though)
NUM_FEATURES = 10


def preprocess(decomposition = False, classification = False, var_feat = False):

    x_train_pr, y_train_pr = x_train, y_train
    x_test_pr, y_test_pr = x_test, y_test
    
    #for the ML models, they can't really handle time series data so I instead averaged the
    #features. This alone might not be enough information so taking the variance of the features might also be good
    if var_feat:
        x_train_pr = np.mean(x_train_pr, axis = 1)
        x_test_pr = np.mean(x_test_pr, axis = 1)
    
    #if we want to perform PCA
    if decomposition:
        #we set the number of features we want to have
        pca = PCA(n_components = NUM_FEATURES)

        if var_feat:

            #pca will look for correlations between our features on the frame level so we reshape the dataset
            #and then fit on it
            x_train_pr = x_train_pr.reshape(x_train_pr.shape[0], 429)
            x_train_pr = pca.fit_transform(x_train_pr).reshape((x_train_pr.shape[0], NUM_FEATURES))
            x_test_pr = x_test_pr.reshape(x_test_pr.shape[0], 429)
            #and then apply it on the test set
            x_test_pr = pca.transform(x_test_pr).reshape((x_test_pr.shape[0], NUM_FEATURES))
        else:
            x_train_pr = x_train_pr.reshape(x_train_pr.shape[0]*10, 429)
            x_train_pr = pca.fit_transform(x_train_pr).reshape((x_train_pr.shape[0]//10, 10, NUM_FEATURES))
            x_test_pr = x_test_pr.reshape(x_test_pr.shape[0]*10, 429)
            x_test_pr = pca.transform(x_test_pr).reshape((x_test_pr.shape[0]//10, 10, NUM_FEATURES))

    #in the emotiw competition, the labels go from 0 to 1 in increments of 1/3 and they treated it as a regression task
    #but you can change the labels so that they're integers and train a classifier instead if you want
    if classification:
        y_train_pr = np.rint(y_train*3).astype(int)
        y_test_pr=  np.rint(y_test*3).astype(int)
    
    return x_train_pr, y_train_pr, x_test_pr, y_test_pr

#For the ML models, I tried out random forest and gradient boosting
x_train_pr, y_train_pr, x_test_pr, y_test_pr = preprocess(decomposition = True, var_feat = True, classification = True)

#For the classifiers, the accuracy wasn't that good. The highest I got was around 42.5% with the gradient boosting
#classifier. The random forest classifier wasn't too far behind though (40%) and I think there are a lot more parameters you 
#can play around with in the GB classifier.

#For the regressors, the random forest regressor did better.  The lowest MSE I was able to get was .088
#which is better than the baseline model for the competition which was .1. And the first place model had an MSE
#of like .05 so this seems pretty promising. Adding the variances of the features will probably really help with this
#And again, gradient boosted trees was close to the random forest regressor (.095).

#max depth is the max number of levels that each tree in the random forest can have. With more levels,
#you run the risk of overfitting but you can counter that by increasing the number of trees. 
clf = RandomForestClassifier(n_estimators = 50, max_depth = 12, random_state = 0)
clf.fit(x_train_pr, y_train_pr)
acc = clf.score(x_train_pr, y_train_pr)
print("Random Forest Classifier Training Accuracy: " + str(acc))
acc = clf.score(x_test_pr, y_test_pr)
print("Random Forest Classifier Test Accuracy: " + str(acc) + "\n")

clf = GradientBoostingClassifier(n_estimators = 3, learning_rate = .4, max_depth= 3, random_state = 0)
clf.fit(x_train_pr, y_train_pr)
acc = clf.score(x_train_pr, y_train_pr)
print("Gradient Boosting Classifier Training Accuracy: " + str(acc))
acc = clf.score(x_test_pr, y_test_pr)
print("Gradient Boosting Classifier Test Accuracy: " + str(acc) + "\n")

x_train_pr, y_train_pr, x_test_pr, y_test_pr = preprocess(decomposition = True, var_feat = True, classification = False)
clf = RandomForestRegressor(max_depth = 3, random_state = 0)
clf.fit(x_train_pr, y_train_pr)
acc = mean_squared_error(y_train_pr, clf.predict(x_train_pr))
print("Random Forest Regressor Training MSE: " + str(acc))
acc = mean_squared_error(y_test_pr, clf.predict(x_test_pr))
print("Random Forest Regressor Test MSE: " + str(acc) + "\n")

clf = GradientBoostingRegressor(n_estimators = 5, learning_rate = .4, max_depth= 3, random_state = 0)
clf.fit(x_train_pr, y_train_pr)
acc = mean_squared_error(y_train_pr, clf.predict(x_train_pr))
print("Gradient Boosting Regressor Training MSE: " + str(acc))
acc = mean_squared_error(y_test_pr, clf.predict(x_test_pr))
print("Gradient Boosting Regressor Test MSE: " + str(acc) + "\n")


#For the DL methods I tried out a CNN and an LSTM. Surprisingly, the CNN does a lot better than 
#the LSTM here even though the LSTM is better suited for time series stuff. With the LSTM, the 
#MSE was around .11 and for the CNN, it got to .106 but it wasn't able to decrease any further. But
#I haven't done any hyperparameter tuning and the the architecture I used is pretty basic so there's
#definitely room for improvement
def DL_Methods(type, classification = False):

    #we perform PCA and modify the labels for classification
    x_train_pr, y_train_pr, x_test_pr, y_test_pr = preprocess(decomposition = True, classification = classification)


    if type == "CNN":
        x_train_pr = x_train_pr.reshape(x_train_pr.shape[0], 10, NUM_FEATURES, 1)
        x_test_pr = x_test_pr.reshape(x_test_pr.shape[0], 10, NUM_FEATURES, 1)
        inputs = keras.Input((10, NUM_FEATURES, 1))
        x = layers.Conv2D(32, 3, activation = "relu")(inputs)
        x = layers.Conv2D(32, 3, activation = "relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(64, (1, 3), activation = "relu")(x)
        x = layers.Conv2D(64, (1,3), activation = "relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.GlobalAveragePooling2D()(x)
        if classification:
            outputs = layers.Dense(units = 4)(x)
        else:
            outputs = layers.Dense(units = 1)(x)
        model = keras.Model(inputs, outputs)

    elif type == "LSTM":
        inputs = keras.Input((10, NUM_FEATURES))
        x = LSTM(1)(inputs)
        x = layers.Dense(16)(x)
        if classification:
            outputs = layers.Dense(4)(x)
        else:
            outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
    
    model.summary()

    if classification:
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = "accuracy"
    else:
        loss = tf.keras.losses.MeanSquaredError()
        metrics = "mae"

    optimizer = tf.keras.optimizers.AdamW()
    model.compile(optimizer=optimizer, loss= tf.keras.losses.MeanSquaredError(), metrics = metrics)

    model.fit(x_train_pr, y_train_pr, epochs = 100, 
         batch_size = 64, validation_data = (x_test_pr, y_test_pr))
 

