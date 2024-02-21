import numpy as np
import pickle
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.environ['GARS_PROJ'])
from util import *

#this sets the sequence length for the input to our model
#with a sequence length of 10 i think there were around 90,000 sequences
#which should be large enough to train on i think but if not we can try 
#decreasing the sequence length
SEQUENCE_LENGTH = 10

#path to the behance dataset files
rec_path = os.path.join(var.GARS_PROJ, "datasets", "Behance")


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
    


print(len(list(user_appreciates.keys())))

for index, user in enumerate(user_appreciates):
    if len(user_appreciates[user]) != SEQUENCE_LENGTH:
        break

item_features = np.load(os.path.join(rec_path, "behance_item_features.npy"))

#with open(os.path.join(rec_path, "behance_item_indices.pkl"), "rb") as f:
#    item_index = pickle.load(f)


behance_dataset = np.zeros((len(user_appreciates), 10, 4096))

for index, user in enumerate(user_appreciates):

    behance_dataset[index] = item_features[user_appreciates[user]]

np.random.shuffle(behance_dataset)


train = behance_dataset[:int(.8*len(behance_dataset))]
val = behance_dataset[int(.8*len(behance_dataset)):int(.9*len(behance_dataset))]
test = behance_dataset[int(.9*len(behance_dataset)):]

