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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ArtRecSystem:
    """Art Recommendation System using Bag of Words Approach"""

    def __init__(
        self,
        metric,
        decay_rate=0.6,
        sample_stage_size=5,
        max_jump=1 / 1000,
        total_iterations=18,
        art_generate=False,  # determines whether art will be outputted
        embed_type="openai"
    ):
        """mimokowski = euclidean distance, cosine = 1 - cosine similarity"""
        #signal.signal(signal.SIGINT, self.signal_handler)
        logging.basicConfig(level=logging.DEBUG)
        gars_path = os.environ["GARS_PROJ"]
        gars_art_path = os.path.join(gars_path, "art_generate")
        self._decay_rate = decay_rate
        self._user_sample_stage_size = sample_stage_size
        if embed_type == "openai":
            self._ndim = 1536
        else:
            self._ndim = 768
        self._matrices = np.zeros((total_iterations + sample_stage_size + 2, 6, self._ndim))

        self._rec_embed_indices = []
        self._cur_embeddings = np.zeros((6, self._ndim))
        self._user_matrix = np.zeros((6, self._ndim))
        self._plaintext_words = np.load(
            f"{gars_art_path}/data_prompts/numpy/plaintext_words.npy"
        )
        self._all_embeddings = np.load(
            f"{gars_art_path}/data_prompts/numpy/{embed_type}/all_embeddings.npy"
        )

        self._total_iterations = total_iterations
        self._metric = metric
        self._iteration = 0
        self._max_jump = 1 / 3
        self._ratings = []
        self._cur_dir = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.mkdir(f"{gars_art_path}/logs/{self._cur_dir}")
        self._cur_dir = f"{gars_art_path}/logs/{self._cur_dir}"
        self_cur_embeddings = np.zeros(
            (6, self._ndim)
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
        self._total_subjects = math.ceil(self._total_subjects/ 2)
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
    def signal_handler(self, sig, frame):
        logging.info("SAVING CURRENT STATE")
        self.save_state()

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
        closest_modifiers = (
            self._nn_modifiers.kneighbors(self._user_matrix[3:])[1][
                :, self.get_val_rand_k(self._total_modifiers, size=3)
            ][0]
            + self._category_indices["descriptive terms"][0]
        )

        self._rec_embed_indices.append(
            [
                int(closest_subject),
                int(closest_artist_and_movement),
                int(closest_medium),
                *closest_modifiers.tolist(),
            ]
        )
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
        logging.debug(f"iteration count:{self._iteration}")
        if self._iteration == self._user_sample_stage_size + self._total_iterations + 2:
            self.save_state()
        self._ratings.append(rating)
        # sampling stage
        if self._iteration < self._user_sample_stage_size:
            rec_words = self.sampling_stage(rating)
            rec_prompt = f"{rec_words[-1]} {rec_words[0]}, {rec_words[1]}, {rec_words[2]}, {rec_words[3]}, {rec_words[4]}"
            self._matrices[self._iteration] += self._user_matrix

            self._iteration += 1
            logging.debug("\n")
            return (
                self.generate_image(rec_prompt),
                rec_prompt,
                rec_words,
            )
        rec_words = self.find_closest(rating)

        rec_prompt = f"{rec_words[-1]} {rec_words[0]}, {rec_words[1]}, {rec_words[2]}, {rec_words[3]}, {rec_words[4]}"
        self._matrices[self._iteration] += self._user_matrix
        self._iteration += 1
        logging.debug("\n") 
        return self.generate_image(rec_prompt), rec_prompt, rec_words


    def create_plots(
        self, plaintext_words, total_embeddings, user_embeddings, rec_indices, title
    ):
        """
        create plotly figures vector spaces
        """

        pca = PCA(n_components=3)
        scaler = StandardScaler()
        all_embeddings = np.concatenate((total_embeddings, user_embeddings))

        x = scaler.fit_transform(all_embeddings)
        z = pca.fit_transform(x)
        len_user = len(user_embeddings)

        z_user = z[-len_user:]
        z_embed = z[:-len_user]
        z_rec_words = z[rec_indices]

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]
            ],
            subplot_titles=("Embedding Space", "Reccomended Words", "User Vector"),
        )
        fig.update_layout(title=title)
        fig.add_trace(
            go.Scatter3d(
                x=z_embed[:, 0],  # First PCA component
                y=z_embed[:, 1],  # Second PCA component
                z=z_embed[:, 2],  # Third PCA component
                mode="markers",
                marker=dict(
                    size=5,
                    color=z[:, 0],  # Color points by the first PCA component
                    colorscale="Viridis",  # Color scale
                    opacity=0.8,
                ),
                text=plaintext_words,  # Labels for each point
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=z_rec_words[:, 0],  # First PCA component
                y=z_rec_words[:, 1],  # Second PCA component
                z=z_rec_words[:, 2],  # Third PCA component
                mode="lines+markers",
                marker=dict(
                    size=5,
                    color=z[:, 0],  # Color points by the first PCA component
                    colorscale="Viridis",  # Color scale
                    opacity=0.8,
                ),
                text=plaintext_words[rec_indices],  # Labels for each point
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter3d(
                x=z_user[:, 0],  # First PCA component
                y=z_user[:, 1],  # Second PCA component
                z=z_user[:, 2],  # Third PCA component
                mode="lines+markers",
                marker=dict(
                    size=5,
                    color=z[:, 0],  # Color points by the first PCA component
                    colorscale="Viridis",  # Color scale
                    opacity=0.8,
                ),
            ),
            row=1,
            col=3,
        )
        sliders = [
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"name-{i}"],
                            dict(
                                mode="immediate",
                                frame=dict(duration=500, redraw=True),
                                fromcurrent=True,
                            ),
                        ],
                        label=f"Frame {i}",
                    )
                    for i, _ in enumerate(fig.frames)
                ],
                transition=dict(duration=300),
                x=0,  # Slider starting position
                y=0,  # Slider vertical position
                currentvalue=dict(
                    font=dict(size=12), prefix="Frame: ", visible=True, xanchor="center"
                ),
                len=1.0,
            )
        ]  # Slider length

        # Add play and stop buttons
        updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                y=1,  # Button vertical position
                x=0.1,  # Button horizontal position
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
                        ],
                    ),
                ],
            )
        ]

        # Update the layout with sliders and buttons
        fig.update_layout(sliders=sliders, updatemenus=updatemenus)
        frames = [
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=z_rec_words[:, 0][:k],  # Similar update for trace 1
                        y=z_rec_words[:, 1][:k],
                        z=z_rec_words[:, 2][:k],
                        mode="lines+markers",
                        marker=dict(size=5, colorscale="Viridis", opacity=0.8),
                    ),
                    go.Scatter3d(
                        x=z_user[:, 0][
                            :k
                        ],  # Different update for trace 2, maybe a different range or pattern
                        y=z_user[:, 1][:k],
                        z=z_user[:, 2][:k],
                        mode="lines+markers",
                        marker=dict(size=5, colorscale="Viridis", opacity=0.8),
                    ),
                ],
                traces=[
                    1,
                    2,
                ],  # Update all three traces, but with different data for trace 2
            )
            for k in range(0, len(rec_indices))
        ]

        fig.frames = frames

        return fig

    def save_plots(self):
        terms_list = [
            self._category_indices["subjects"],
            self._category_indices["artists"],
            self._category_indices["mediums"],
            self._category_indices["descriptive terms"],
            self._category_indices["descriptive terms"],
            self._category_indices["descriptive terms"],
        ]
        titles = [
            "Subjects",
            "style-art movements",
            "Medium",
            "Modifier 0",
            "Modifier 1",
            "Modifier 2",
        ]
        numpy_list = np.array(self._rec_embed_indices)
        for i in range(len(terms_list)):
            fig = self.create_plots(
                plaintext_words=self._plaintext_words[ terms_list[i][0] : terms_list[i][1] + 1],
                total_embeddings=self._all_embeddings[
                    terms_list[i][0] : terms_list[i][1] + 1
                ],
                user_embeddings=self._matrices[:, i],
                rec_indices=numpy_list[:, i] - terms_list[i][0],
                title=titles[i],
            )
            fig.write_html(f"{self._cur_dir}/{titles[i]}.html")

        return True

    # animations

    def save_state(self):
        print("SAVING CURRENT STATE")
        with open(f"{self._cur_dir}/rec_embed_indices.json", "w") as fp:
            json.dump(self._rec_embed_indices, fp)

        with open(f"{self._cur_dir}/ratings.json", "w") as fp:
            json.dump(self._ratings, fp, indent=4)

        np.save(f"{self._cur_dir}/user_matrices.npy", self._matrices)

        self.save_plots()

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

        self._rec_embed_indices.append(
            [
                int(subject),
                int(artists_and_movements),
                int(mediums),
                *modifiers.tolist(),
            ]
        )
        indices = np.concatenate((subject, artists_and_movements, mediums, modifiers))
        self._cur_embeddings = self._all_embeddings[indices]

        return self._plaintext_words[indices]

        # rating = input("rate from -1 to 1: ")
        # self._user_matrix += float(rating) * self._all_embeddings[indices]

# change to main to get embeddings if they are not there
if __name__ == "__main2__":
    if len(sys.argv) > 1:
        argument = sys.argv[1]
    
    if argument == "openai":
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')
        embed_path = "openai"
    else:
        embed_path = "sbert"
    def get_embedding_open_ai(text, model="text-embedding-3-small"):
        print(text)
        return openai.embeddings.create(input = [text], model=model).data[0].embedding
    
    def encode_embeddings_openai(words):
        return  np.array([get_embedding_open_ai(word) for word in words])

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

    if embed_path == "openai":
        subject_embeddings = encode_embeddings_openai(subjects)
        art_mediums_embeddings = encode_embeddings_openai(art_mediums)
        artists_and_movements_embeddings = encode_embeddings_openai(artists_and_movements)
        descriptive_words_embeddings = encode_embeddings_openai(descriptive_words)
    else:
        model = SentenceTransformer("all-mpnet-base-v2")
        subject_embeddings = model.encode(subjects)
        art_mediums_embeddings = model.encode(art_mediums)
        artists_and_movements_embeddings = model.encode(artists_and_movements)
        descriptive_words_embeddings = model.encode(descriptive_words)

    np.save(
        f"{gars_art_path}/data_prompts/numpy/subject_embeddings.npy", subject_embeddings
    )
    np.save(
        f"{gars_art_path}/data_prompts/numpy/{embed_path}/art_mediums_embeddings.npy",
        art_mediums_embeddings,
    )
    np.save(
        f"{gars_art_path}/data_prompts/numpy/{embed_path}/artists_embeddings.npy",
        artists_and_movements,
    )
    np.save(
        f"{gars_art_path}/data_prompts/numpy/{embed_path}/modifiers_embeddings.npy",
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
        f"{gars_art_path}/data_prompts/numpy/{embed_path}/all_embeddings.npy",
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
    rec = ArtRecSystem(metric="cosine", embed_type="openai")#
    #breakpoint()
    while 1:
        rating = input("get rating:")

        # [descrptive term] [subject] [style] [medium] [modifer] [modifier]
        rec_img, rec_prompt, rec_words = rec(rating)
        
        print(rec_prompt)

#test_system()

