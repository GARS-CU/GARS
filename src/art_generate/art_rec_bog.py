from transformers import BertModel, BertTokenizer
from sklearn.neighbors import BallTree
# Load the tokenizer and model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from numpy import linalg as LA
import logging
import json 
class ArtRecSystem:
    """Art Recommendation System using Bag of Words Approach"""

    def __init__(self, decay_rate=0.6, sample_size=10, sample_stage_size=5):
        # loading bert
       # self.__tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #self.__model = BertModel.from_pretrained("bert-base-uncased")

        self._sample_size = sample_size
        self._user_sample_stage_size = sample_stage_size
        self._matrices = np.zeros((40, 6, 768))
        self._user_matrix = np.zeros((6, 768))
        self._plaintext_words = np.load("data/numpy/plaintext_words.npy")
        self._all_embeddings = np.load("data/numpy/all_embeddings.npy")

        with open('categories.json', 'r') as file:
            self._category_indexes = json.load(file)

            #ball tree for finding most similar items   
        self._subject_ball_tree = BallTree(self._all_embeddings[0:self._category_indexes["mediums"][0]])
        self._mediums_ball_tree = BallTree(self._all_embeddings[self._category_indexes["mediums"][0]:self._category_indexes["artists"][0]])
        self._artists_and_movements_ball_tree =BallTree(self._all_embeddings[self._category_indexes["artists"][0]:self._category_indexes["descriptive terms"][0]])
        self._modifiers_ball_tree = BallTree(self._all_embeddings[self._category_indexes["descriptive terms"][0]:])
        self._decay_rate = 0.6


#    def get_embed(self, text):
   #     encoded_input = self._tokenizer(text, return_tensors="pt")
     #   with torch.no_grad():
     #       output = self._model(**encoded_input)
       # return output.last_hidden_state.squeeze()

    def find_closest(self):
        rec_matr = np.zeros((len(self._user_matrix), 768))

        closest_subject = np.array(self._subject_ball_tree.query([self._user_matrix[0]], k = 1))[1].astype(int)


        closest_medium = self._category_indexes["mediums"][0] + np.array(self._mediums_ball_tree.query([self._user_matrix[2]], k = 1))[1].astype(int)
        
        #offset of five because first five are subjects
        closest_artist_and_movement = self._category_indexes["artists"][0] + np.array(self._artists_and_movements_ball_tree.query([self._user_matrix[1]], k = 1))[1].astype(int)


        

        closest_modifiers = self._category_indexes["descriptive terms"][0] + np.array(self._modifiers_ball_tree.query(self._user_matrix[3:], k = 1))[1].astype(int)

     

        indices = np.concatenate((closest_subject, closest_artist_and_movement, closest_medium, closest_modifiers)).squeeze()

        return self._all_embeddings[indices], self._plaintext_words[indices]
        
                                 

    def __call__(self):

        embeddings, words = self._find_closest()


        print(f"here are chosen words {words}")
        rating = input("rate from -1 to 1:")
        self._user_matrix += float(rating) * embeddings
        self._user_matrix *= self._decay_rate

        





    

    def sampling_stage(self):
    #cold starting rec system

        for i in range(self._user_sample_stage_size):

            modifiers = np.random.choice(range(79, 484), size = 3)
            subject = np.random.choice(range(0, 5), size = 1)
            artists_and_movements = np.random.choice(range(28, 143), size = 1)
            mediums = np.random.choice(range(6, 28), size = 1)
            indices = np.concatenate((subject, artists_and_movements, mediums, modifiers))
            print(self._plaintext_words[indices])
            rating = input("rate from -1 to 1: ")
            self._user_matrix += float(rating) * self._all_embeddings[indices]
z = ArtRecSystem()


if __name__ == "__test__":
    from sentence_transformers import SentenceTransformer
     ## [DESCRIPTIVE TERM] OF [SUBJECT] [?Location] [MEDIUM] [MODIFIERS] [2] [3]
    
    with open('./data/categories/artists.txt') as f:
        artists = f.read().split("\n")
   

    #getting descriptive words 
    with open('data/categories/subjects.txt') as f:
        subjects = f.read().split("\n")

    with open("data/open-prompts/modifiers/art/descriptive terms.txt", "r") as f:
        descriptive_words = f.read().split("\n")
    #from #https://www.smore.com/n/st133-art-vocabulary-adjectives
    with open("data/categories/descriptive_words_more.txt") as f:
        descriptive_words.extend(f.read().split("\n"))

    with open("data/open-prompts/modifiers/art/art movements.txt") as f:
        art_movements = f.read().split("\n")

    artists_and_movements = np.concatenate((artists, art_movements))
    with open("data/categories/art_mediums.txt") as f:
        art_mediums = f.read().split("\n")
    model = SentenceTransformer("all-mpnet-base-v2")

    subject_embeddings = model.encode(subjects)
    art_mediums_embeddings = model.encode(art_mediums)
    artists_and_movements_embeddings = model.encode(artists_and_movements)
    descriptive_words_embeddings = model.encode(descriptive_words)

    np.save("data/numpy/subject_embeddings.npy", subject_embeddings)
    np.save("data/numpy/art_mediums_embeddings.npy", art_mediums_embeddings)
    np.save("data/numpy/artists_embeddings.npy", artists_and_movements)
    np.save("data/numpy/modifiers_embeddings.npy",descriptive_words_embeddings)

    cur_val = len(subject_embeddings) 
    
    #create a dictionary of proper indexes of each category
    category_indexes = {}
    
    print(f"subjects 0 - {len(subject_embeddings) - 1}")
    category_indexes["subjects"] = [0, len(subject_embeddings) - 1]

    print(f"mediums {cur_val} - {cur_val + len(art_mediums_embeddings) - 1}" ) #mediums
    category_indexes["mediums"] = [cur_val,len(subject_embeddings) + len(art_mediums_embeddings) - 1]
    
    cur_val += len(art_mediums_embeddings)
    print(f"artists {cur_val} - {cur_val +len(artists_and_movements_embeddings)  - 1}")  #artists
    category_indexes["artists"] = [cur_val, cur_val + len(artists_and_movements_embeddings) - 1]

    cur_val += len(artists_and_movements_embeddings)
    print(f"descriptive term {cur_val} - {cur_val +  len(descriptive_words_embeddings) - 1}") #descriptive terms
    category_indexes["descriptive terms"] = [cur_val, cur_val + len(descriptive_words_embeddings) + 1]


    np.save("data/numpy/all_embeddings.npy", np.concatenate((subject_embeddings,  art_mediums_embeddings,  artists_and_movements_embeddings, descriptive_words_embeddings)))
    plaintext_words = np.concatenate((subjects, art_mediums, artists_and_movements, descriptive_words))
    np.save("data/numpy/plaintext_words.npy", plaintext_words)

    file_name = "categories.json"
    with open(file_name, 'w') as file:
        json.dump(category_indexes, file, indent=4)

    


"""def cosine_sim(a, b):
    return np.dot(a, b) / (LA.norm(a) * LA.norm(b))


text = "artifical intelligence"
text2 = "human intelligence"
rec_model = ArtRecSystem()
embeddings_p = rec_model.get_embed(text)
embeddings_p2 = rec_model.get_embed(text2)
# The last hidden state is the sequence of hidden states of the last layer of the model.
# Obtaining the embeddings for each token in the sentence

embeddings = np.load("embeddings.npy", mmap_mode="r")
avg1 = embeddings_p.mean(axis=0)
avg2 = embeddings_p2.mean(axis=0)
breakpoint()
z = torch.cosine_similarity(avg1.reshape(1, -1), avg2.reshape(1, -1))
x = cosine_sim(avg1, avg2)
from transformers import pipeline

text_pipe = pipeline("text-generation", model="LykosAI/GPT-Prompt-Expansion-Fooocus-v2")

prompt = "ocean in style of van gogh"
extended_prompt = text_pipe(prompt)
print(extended_prompt)


"""
