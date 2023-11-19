import os
import numpy as np
from PIL import Image
from tqdm import tqdm


classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
path_dataset = os.path.abspath("/zooper2/gars/datasets/FER2013")

def load_data(subset):
    pathname = os.path.join(path_dataset, subset)
    data = np.zeros((1, 48, 48))
    labels = []

    for index in range(len(classes)):
        class_path = os.path.join(pathname, classes[index])
        for path in tqdm(os.listdir(class_path)):

            img = Image.open(os.path.join(class_path, path))
            data = np.concatenate((data, np.asarray(img).reshape(1, 48, 48)))
            labels.append(index)
    
    return data[1:], np.asarray(labels)

x_train, y_train = load_data("train")

x_test, y_test = load_data("test")

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

