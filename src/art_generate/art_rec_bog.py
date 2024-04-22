from sklearn.neighbors import NearestNeighbors
import numpy as np
from diffusers import AutoPipelineForText2Image
import torch
import logging
import json
import os
import sys
import signal
import math
from datetime import datetime





class ArtRecSystem:
    """Art Recommendation System using Bag of Words Approach"""

    def __init__(
        self,
        metric,
        decay_rate=0.6,
        sample_size=10,
        sample_stage_size=5,
        max_jump=1 / 1000,
        total_iterations=20,
        art_generate=False,  # determines whether art will be outputted
    ):
        """mimokowski = euclidean distance, cosine = 1 - cosine similarity"""
        signal.signal(signal.SIGINT, self.signal_handler)
        logging.basicConfig(level=logging.DEBUG)
        gars_path = os.environ["GARS_PROJ"]
        gars_art_path = os.path.join(gars_path, "art_generate")
        self._sample_size = sample_size
        self._decay_rate = decay_rate
        self._user_sample_stage_size = sample_stage_size
        self._matrices = np.zeros((total_iterations, 6, 768))
        
        self._rec_embed_indices = []
        self._cur_embeddings = np.zeros(( 6, 768))
        self._user_matrix = np.zeros((6, 768))
        self._plaintext_words = np.load(
            f"{gars_art_path}/data_prompts/numpy/plaintext_words.npy"
        )
        self._all_embeddings = np.load(
            f"{gars_art_path}/data_prompts/numpy/all_embeddings.npy"
        )
        self._total_iterations = total_iterations
        self._metric = metric
        self._iteration = 0
        self._max_jump = 1 / 3
        self._ratings = []
        self._cur_dir = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir(self._cur_dir)
        self_cur_embeddings = np.zeros(
            (6, 768)
        )  # current embeddings of words that were recommended
        if metric == "cosine":
            self.scoring = self.moving_cosine_dist
        else:
            self.scoring = self.moving_euclidean

        with open(f"{gars_art_path}/categories.json", "r") as file:
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

        self._total_subjects = (
            self._category_indices["subjects"][1]
            - self._category_indices["subjects"][0]
            + 1
        )
        self._nn_subjects = NearestNeighbors(
            n_neighbors=self._total_subjects, metric=metric
        )
        self._nn_subjects.fit(
            self._all_embeddings[0 : self._category_indices["mediums"][0]],
            range(0, self._category_indices["mediums"][0]),
        )  # currently recommended words

        self._total_mediums = (
            self._category_indices["mediums"][1]
            - self._category_indices["mediums"][0]
            + 1
        )
        self._nn_mediums = NearestNeighbors(
            n_neighbors=self._total_mediums, metric=metric
        )
        self._nn_mediums.fit(
            self._all_embeddings[
                self._category_indices["mediums"][0] : self._category_indices[
                    "artists"
                ][0]
            ]
        )
        self._total_artists_movements = (
            self._category_indices["artists"][1]
            - self._category_indices["artists"][0]
            + 1
        )
        self._nn_artists_and_movements = NearestNeighbors(
            n_neighbors=self._total_artists_movements, metric=metric
        )
        self._nn_artists_and_movements.fit(
            self._all_embeddings[
                self._category_indices["artists"][0] : self._category_indices[
                    "descriptive terms"
                ][0]
            ]
        )

        self._total_modifiers = (
            self._category_indices["descriptive terms"][1]
            - self._category_indices["descriptive terms"][0]
            + 1
        )

        self._nn_modifiers = NearestNeighbors(
            n_neighbors=self._total_modifiers, metric=metric
        )
        self._nn_modifiers.fit(
            self._all_embeddings[self._category_indices["descriptive terms"][0] :]
        )
        gars_path = os.environ["GARS_PROJ"]
        self._gars_art_path = os.path.join(gars_path, "art_generate")
    # cc    breakpoint()
    # potentially adjust decay rate
    def signal_handler(self,sig, frame):
        logging.info("SAVING CURRENT STATE")
        self.save_state()
        sys.exit(0)
       

    def get_val_rand_k(self, total, size=None):
        """gets random top k value lowering k as iterations increase"""
        k = self.exp_decay(total)
        logging.debug(f"k = {k}")
        return np.random.randint(k, size=size)

    def find_closest(self, rating):
        """finds the closest embedding vector for a specific user vector
        there is no sampling implemented yet"""
        # rec_matr = np.zeros((len(self._user_matrix), 768))
        # last sample stage recomendation
        self.scoring(rating, self._cur_embeddings)

        closest_subject = self._nn_subjects.kneighbors([self._user_matrix[0]])[1][
            :, self.get_val_rand_k(self._total_subjects)
        ]
        closest_artist_and_movement = (
            self._nn_artists_and_movements.kneighbors([self._user_matrix[1]])[1][
                :, self.get_val_rand_k(self._total_artists_movements)
            ]
            + self._category_indices["artists"][0]
        )
        closest_medium = (
            self._nn_mediums.kneighbors([self._user_matrix[2]])[1][
                :, self.get_val_rand_k(self._total_mediums)
            ]
            + self._category_indices["mediums"][0]
        )
        closest_modifiers = self._nn_modifiers.kneighbors(self._user_matrix[3:])[1][:,self.get_val_rand_k(self._total_modifiers, size=3)][0] + self._category_indices["descriptive terms"][0]

        self._rec_embed_indices.append([int(closest_subject), int(closest_artist_and_movement), int(closest_medium),  closest_modifiers.tolist()])
        indices = np.concatenate(
            (
                closest_subject,
                closest_artist_and_movement,
                closest_medium,
                closest_modifiers,
            )
        ).squeeze()
        # breakpoint()

        self._user_matrix *= self._decay_rate

        self._cur_embeddings = self._all_embeddings[indices]

        return self._plaintext_words[indices]

    def exp_decay(self, total):
        """decay  the amount of nearest neighbors so it approaches 1 after final iteration"""
        self._non_sample_iter = self._iteration - self._user_sample_stage_size
        r = (1 / total) ** (1 / (self._total_iterations))
        return math.ceil(total * r ** (self._non_sample_iter))

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
        
        self._ratings.append(rating)
        # sampling stage
        if self._iteration < self._user_sample_stage_size:
            rec_words = self.sampling_stage(rating)
            rec_prompt = f"{rec_words[-1]} {rec_words[0]}, {rec_words[1]}, {rec_words[2]}, {rec_words[3]}, {rec_words[4]}"
            self._matrices[self._iteration] += self._user_matrix

            self._iteration += 1
            
            return (
                self.generate_image(rec_prompt),
                rec_prompt,
                rec_words,
            )
        rec_words = self.find_closest(rating)
        
        rec_prompt = f"{rec_words[-1]} {rec_words[0]}, {rec_words[1]}, {rec_words[2]}, {rec_words[3]}, {rec_words[4]}"
        self._iteration += 1
        return self.generate_image(rec_prompt), rec_prompt, rec_words
    def save_state(self):
            with open(f"{self._cur_dir}/rec_embed_indices.json", "w") as fp:
                json.dump(self._rec_embed_indices, fp)

            with open(f"{self._cur_dir}/ratings.json", "w") as fp:
                json.dump(self._ratings, fp,  indent = 4)

            np.save(f"{self._cur_dir}/user_matrices.npy", self._matrices)
            


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
            size=1
        )
        mediums = np.random.choice(
            range(
                self._category_indices["mediums"][0],
                self._category_indices["artists"][0],
            ),
            size=1,
        )
        self._rec_embed_indices.append([int(subject), int(artists_and_movements), int(mediums),  modifiers.tolist()])
        indices = np.concatenate((subject, artists_and_movements, mediums, modifiers))
        self._cur_embeddings = self._all_embeddings[indices]

        return self._plaintext_words[indices]
        

       
        # rating = input("rate from -1 to 1: ")
        # self._user_matrix += float(rating) * self._all_embeddings[indices]

# change to main to get embeddings if they are not there
if __name__ == "__main__":
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
        cur_val + len(art_mediums_embeddings) - 1,
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
    file_name = f"{gars_art_path}/categories.json"
    with open(file_name, "w") as file:
        json.dump(category_indices, file, indent=4)


def test_system():
    rec = ArtRecSystem(metric="cosine")
    while 1:
        rating = input("get rating:")

        # [descrptive term] [subject] [style] [medium] [modifer] [modifier]
        rec_img, rec_prompt, rec_words = rec(rating)
        print(rec_prompt)
#test_system()

