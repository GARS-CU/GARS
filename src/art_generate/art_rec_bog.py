from transformers import BertModel, BertTokenizer

from sklearn.neighbors import BallTree

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy import linalg as LA
import logging
from sklearn.neighbors import KNeighborsClassifier
import json


class ArtRecSystem:
    """Art Recommendation System using Bag of Words Approach"""

    def __init__(
        self,
        metric,
        decay_rate=0.6,
        sample_size=10,
        sample_stage_size=5,
    ):
        """mimokowski = euclidean distance, cosine = 1 - cosine similarity"""
        self._sample_size = sample_size
        self._user_sample_stage_size = sample_stage_size
        self._matrices = np.zeros((40, 6, 768))
        self._user_matrix = np.zeros((6, 768))
        self._cur_embeddings = np.zeros((6, 768))  # currently recommended words
        self._plaintext_words = np.load("data/numpy/plaintext_words.npy")
        self._all_embeddings = np.load("data/numpy/all_embeddings.npy")
        self._metric = metric
        self._iteration = 0
        self_cur_embeddings = np.zeros(
            (6, 768)
        )  # current embeddings of words that were recommended

        if metric == "cosine":
            self.scoring = self.moving_cosine_dist
        else:
            self.scoring = self.moving_euclidean

        with open("categories.json", "r") as file:
            self._category_indices = json.load(file)

        self.knn_subjects = KNeighborsClassifier(n_neighbors=5, metric=metric)
        self.knn_subjects.fit(
            self._all_embeddings[0 : self._category_indices["mediums"][0]],
            range(0, self._category_indices["mediums"][0]),
        )  # currently recommended words

        self.knn_mediums = KNeighborsClassifier(n_neighbors=5, metric=metric)
        self.knn_mediums.fit(
            self._all_embeddings[
                self._category_indices["mediums"][0] : self._category_indices[
                    "artists"
                ][0]
            ],
            range(
                self._category_indices["mediums"][0],
                self._category_indices["artists"][0],
            ),
        )

        self.knn_artists_and_movements = KNeighborsClassifier(
            n_neighbors=5, metric=metric
        )
        self.knn_artists_and_movements.fit(
            self._all_embeddings[
                self._category_indices["artists"][0] : self._category_indices[
                    "descriptive terms"
                ][0]
            ],
            range(
                self._category_indices["artists"][0],
                self._category_indices["descriptive terms"][0],
            ),
        )

        self.knn_modifiers = KNeighborsClassifier(n_neighbors=5, metric=metric)
        self.knn_modifiers.fit(
            self._all_embeddings[self._category_indices["descriptive terms"][0] :],
            range(
                self._category_indices["descriptive terms"][0],
                self._category_indices["descriptive terms"][1] + 1,
            ),
        )

    # cc    breakpoint()
    # potentially adjust decay rate

    def find_closest(self, rating):
        """finds the closest embedding vector for a specific user vector
        there is no sampling implemented yet"""
        # rec_matr = np.zeros((len(self._user_matrix), 768))
        # last sample stage recomendation
        self.scoring(rating, self._cur_embeddings)
        if self._iteration != self._user_sample_stage_size:
            self._user_matrix *= self._decay_rate

        closest_subject = self.knn_subjects.kneighbors([self._user_matrix[0]])[1]
        closest_artist_and_movement = self.knn_artists_and_movements.kneighbors(
            [self._user_matrix[1]]
        )[1]
        closest_medium = self.knn_mediums.kneighbors([self._user_matrix[2]])[1]
        closest_modifiers = self.knn_modifiers.kneighbors(self._user_matrix[3:])[1]

        indices = np.concatenate(
            (
                closest_subject,
                closest_artist_and_movement,
                closest_medium,
                closest_modifiers,
            )
        ).squeeze()
        # breakpoint()
        self._cur_embeddings = self._all_embeddings[indices]

        return self._plaintext_words[indices]

    def moving_euclidean(self, rating, embeddings):
        self._user_matrix = (embeddings - self._user_matrix) * float(rating)

    def moving_cosine_dist(self, rating, embeddings):
        self._user_matrix += float(rating) * embeddings

    def __call__(self, rating=None) -> str:
        """function to get recommendation given a rating"""

        # sampling stage
        if self._iteration < self._user_sample_stage_size:
            self._iteration += 1
            return self.sampling_stage(rating)

        return self.find_closest(rating)

        return None
        embeddings, words = self.find_closest()

        print(f"here are chosen words {words}")
        rating = input("rate from -1 to 1:")
        self._user_matrix += float(rating) * embeddings
        self._user_matrix *= self._decay_rate

    def sampling_stage(self, rating):
        # cold starting rec system
        self.scoring(rating, self._cur_embeddings)

        modifiers = np.random.choice(
            range(
                self._category_indices["descriptive terms"][0],
                self._category_indices["descriptive terms"][1] + 1,
            ),
            size=3,
        )
        subject = np.random.choice(
            range(0, self._category_indices["mediums"][0]), size=1
        )
        artists_and_movements = np.random.choice(
            range(
                self._category_indices["artists"][0],
                self._category_indices["descriptive terms"][0],
            ),
            size=1,
        )
        mediums = np.random.choice(
            range(
                self._category_indices["mediums"][0],
                self._category_indices["artists"][0],
            ),
            size=1,
        )
        indices = np.concatenate((subject, artists_and_movements, mediums, modifiers))

        self._cur_embeddings = self._all_embeddings[indices]

        return self._plaintext_words[indices]

        # rating = input("rate from -1 to 1: ")
        # self._user_matrix += float(rating) * self._all_embeddings[indices]


def test_system():
    rec = ArtRecSystem(metric="cosine")
    while 1:
        rating = input("get rating:")

        # [descrptive term] [subject] [style] [medium] [modifer] [modifier]
        rec_words = rec(rating)
        print(
            f"prompt {rec_words[-1]} {rec_words[0]}, {rec_words[1]}, {rec_words[2]}, {rec_words[3]}, {rec_words[4]}"
        )


test_system()


# change to main to get embeddings if they are not there
if __name__ == "__tmain__":
    from sentence_transformers import SentenceTransformer

    ## [DESCRIPTIVE TERM] OF [SUBJECT] [?Location] [MEDIUM] [MODIFIERS] [2] [3]

    with open("./data/categories/artists.txt") as f:
        artists = f.read().split("\n")

    # getting descriptive words
    with open("data/categories/subjects.txt") as f:
        subjects = f.read().split("\n")

    with open("data/open-prompts/modifiers/art/descriptive terms.txt", "r") as f:
        descriptive_words = f.read().split("\n")
    # from #https://www.smore.com/n/st133-art-vocabulary-adjectives
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
    np.save("data/numpy/modifiers_embeddings.npy", descriptive_words_embeddings)

    cur_val = len(subject_embeddings)

    # create a dictionary of proper indices of each category
    category_indices = {}

    print(f"subjects 0 - {len(subject_embeddings) - 1}")
    category_indices["subjects"] = [0, len(subject_embeddings) - 1]

    print(f"mediums {cur_val} - {cur_val + len(art_mediums_embeddings) - 1}")  # mediums
    category_indices["mediums"] = [
        cur_val,
        len(subject_embeddings) + len(art_mediums_embeddings) - 1,
    ]

    cur_val += len(art_mediums_embeddings)
    print(
        f"artists {cur_val} - {cur_val +len(artists_and_movements_embeddings)  - 1}"
    )  # artists
    category_indices["artists"] = [
        cur_val,
        cur_val + len(artists_and_movements_embeddings) - 1,
    ]

    cur_val += len(artists_and_movements_embeddings)
    print(
        f"descriptive term {cur_val} - {cur_val +  len(descriptive_words_embeddings) - 1}"
    )  # descriptive terms
    category_indices["descriptive terms"] = [
        cur_val,
        cur_val + len(descriptive_words_embeddings) - 1,
    ]

    np.save(
        "data/numpy/all_embeddings.npy",
        np.concatenate(
            (
                subject_embeddings,
                art_mediums_embeddings,
                artists_and_movements_embeddings,
                descriptive_words_embeddings,
            )
        ),
    )
    plaintext_words = np.concatenate(
        (subjects, art_mediums, artists_and_movements, descriptive_words)
    )
    np.save("data/numpy/plaintext_words.npy", plaintext_words)
    file_name = "categories.json"
    with open(file_name, "w") as file:
        json.dump(category_indices, file, indent=4)


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
