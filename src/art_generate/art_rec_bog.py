from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
import logging
import json
import os
import sys


class ArtRecSystem:
    """Art Recommendation System using Bag of Words Approach"""

    def __init__(
        self,
        metric,
        decay_rate=0.6,
        sample_size=10,
        sample_stage_size=5,
        max_jump=1 / 3,
        art_generate=False,  # determines whether art will be outputted
    ):
        """mimokowski = euclidean distance, cosine = 1 - cosine similarity"""

        gars_path = os.environ["GARS_PROJ"]
        gars_art_path = os.path.join(gars_path, "art_generate")

        self._sample_size = sample_size
        self._user_sample_stage_size = sample_stage_size
        self._matrices = np.zeros((40, 6, 768))
        self._user_matrix = np.zeros((6, 768))
        self._cur_embeddings = np.zeros((6, 768))  # currently recommended words
        self._plaintext_words = np.load(
            f"{gars_art_path}/data_prompts/numpy/plaintext_words.npy"
        )
        self._all_embeddings = np.load(
            f"{gars_art_path}/data_prompts/numpy/all_embeddings.npy"
        )
        self._metric = metric
        self._iteration = 0
        self._max_jump = 1 / 3
        self_cur_embeddings = np.zeros(
            (6, 768)
        )  # current embeddings of words that were recommended
        if metric == "cosine":
            self.scoring = self.moving_cosine_dist
        else:
            self.scoring = self.moving_euclidean

        with open("categories.json", "r") as file:
            self._category_indices = json.load(file)

        if art_generate:
            # for now and testing sdxl-turbo, for actual one bytedance model, stable cascade still not working
            self.sdxl_turbo = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
            )
            self.sdxl_turbo.to("cuda")
            self.art_generate = True
        else:
            self.art_generate = False

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
        if self._iteration != self._user_sample_stage_size:
            self._user_matrix *= self._decay_rate

        self._cur_embeddings = self._all_embeddings[indices]

        return self._plaintext_words[indices]

    def generate_image(self, prompt):
        """generates an image given a prompt"""

        if self.art_generate:
            image = self.sdxl_turbo(
                prompt=prompt, num_inference_steps=1, guidance_scale=0.0
            ).images[0]
        else:
            image = -1

        return image

    def max_jump(self):
        """Can be used to make more dynamic jumping"""
        return self._max_jump

    def moving_euclidean(self, rating, embeddings):
        self._user_matrix += (
            (embeddings - self._user_matrix) * float(rating) * self.max_jump
        )

    def moving_cosine_dist(self, rating, embeddings):
        self._user_matrix += float(rating) * embeddings

    def __call__(self, rating=0):
        """function to get recommendation given a rating"""

        # sampling stage
        if self._iteration < self._user_sample_stage_size:
            self._iteration += 1
            rec_words = self.sampling_stage(rating)
            rec_prompt = f"{rec_words[-1]} {rec_words[0]}, {rec_words[1]}, {rec_words[2]}, {rec_words[3]}, {rec_words[4]}"

            return (
                self.generate_image(rec_prompt),
                rec_prompt,
                rec_words,
            )
        self._iteration += 1
        rec_words = self.find_closest(rating)
        rec_prompt = f"{rec_words[-1]} {rec_words[0]}, {rec_words[1]}, {rec_words[2]}, {rec_words[3]}, {rec_words[4]}"

        return self.generate_image(rec_prompt), rec_prompt, rec_words

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
        rec_img, rec_prompt, rec_image = rec(rating)
        print(rec_prompt)


# test_system()
# change to main to get embeddings if they are not there
if __name__ == "__tmain__":
    from sentence_transformers import SentenceTransformer

    gars_path = os.environ["GARS_PROJ"]
    gars_art_path = os.path.join(gars_path, "art_generate")

    ## [DESCRIPTIVE TERM] OF [SUBJECT] [?Location] [MEDIUM] [MODIFIERS] [2] [3]

    with open(f"{gars_art_path}/data_prompts/categories/artists.txt") as f:
        artists = f.read().split("\n")

    # getting descriptive words
    with open(f"{gars_art_path}/data_prompts/categories/subjects.txt") as f:
        subjects = f.read().split("\n")

    with open(
        f"{gars_art_path}/data_prompts/open-prompts/modifiers/art/descriptive terms.txt",
        "r",
    ) as f:
        descriptive_words = f.read().split("\n")
    # from #https://www.smore.com/n/st133-art-vocabulary-adjectives
    with open(
        f"{gars_art_path}/data_prompts/categories/descriptive_words_more.txt"
    ) as f:
        descriptive_words.extend(f.read().split("\n"))

    with open(
        f"{gars_art_path}/data_prompts/open-prompts/modifiers/art/art movements.txt"
    ) as f:
        art_movements = f.read().split("\n")

    artists_and_movements = np.concatenate((artists, art_movements))
    with open(f"{gars_art_path}/data_prompts/categories/art_mediums.txt") as f:
        art_mediums = f.read().split("\n")
    model = SentenceTransformer("all-mpnet-base-v2")

    subject_embeddings = model.encode(subjects)
    art_mediums_embeddings = model.encode(art_mediums)
    artists_and_movements_embeddings = model.encode(artists_and_movements)
    descriptive_words_embeddings = model.encode(descriptive_words)

    np.save(
        f"{gars_art_path}/data_prompts/numpy/subject_embeddings.npy", subject_embeddings
    )
    np.save(
        f"{gars_art_path}/data_prompts/numpy/art_mediums_embeddings.npy",
        art_mediums_embeddings,
    )
    np.save(
        f"{gars_art_path}/data_prompts/numpy/artists_embeddings.npy",
        artists_and_movements,
    )
    np.save(
        f"{gars_art_path}/data_prompts/numpy/modifiers_embeddings.npy",
        descriptive_words_embeddings,
    )

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
        f"{gars_art_path}/data_prompts/numpy/all_embeddings.npy",
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
    np.save(f"{gars_art_path}/data_prompts/numpy/plaintext_words.npy", plaintext_words)
    file_name = "categories.json"
    with open(file_name, "w") as file:
        json.dump(category_indices, file, indent=4)
