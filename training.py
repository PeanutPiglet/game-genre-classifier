from PIL import Image
import numpy as np
import os
import time
import json
from common_types import *


TRAINING_IMAGE_W = 128
TRAINING_IMAGE_H = 64


def train(network: Network, epochs: int):
    begin_time = time.time()
    print(f"starting training with {epochs} epochs at {time.asctime()}")

    i = 0
    while i < epochs:
        ttime = time.time()
        train_epoch(network)
        i += 1
        print(f"epoch {i} / {epochs}  --  taken {int(time.time() - ttime)} seconds")

    print(f"ending at {time.asctime()}  --  taken {int(time.time() - begin_time)} seconds")
    return


def train_epoch(network: Network):
    expected_image_size = TRAINING_IMAGE_W * TRAINING_IMAGE_H * 3
    batches = get_batches()
    n = len(batches)
    i = 0
    while i < n:
        batch = batches[i]
        manifest = get_manifest(batch)
        n_entries = len(manifest)
        print(f"{n_entries} entries")
        wrong_size_counter = 0
        j = 0
        last_print_j = j
        for appid in manifest:
            image = load_image(os.path.join(batch, f"{appid}.jpg"))
            if len(image) == expected_image_size:
                res = network.feedforward(image)
                actual = transform_genres_to_vector(manifest[appid])
                costs = 2 * (res - actual)
                network.backprop(costs)
                network.sgd()
            else:
                wrong_size_counter += 1

            j += 1
            if j - last_print_j >= 100 or j == n_entries:
                last_print_j = j
                print(f"{j} / {n_entries}")
        i += 1
        print(f"batch {i} / {n}  --  skipped {wrong_size_counter} images of wrong size")
    return


def get_manifest(batch: str) -> dict:
    with open(os.path.join(batch, 'appdetails.json'), 'r') as f:
        data = json.load(f)
        manifest = {appid: data[appid]['genres'] for appid in data}
        return manifest


def get_batches() -> list[str]:
    directory = "data/"
    batches = [os.path.join(directory, x) for x in os.listdir(directory) if os.path.isdir(os.path.join(directory, x))]
    return batches


def get_batch(batch: str) -> list[str]:
    paths = [os.path.join(batch, x) for x in os.listdir(batch) if x[-4:] == ".jpg"]
    return paths


def transform_genres_to_vector(genres: list[int]):
    # assert len(genres) > 0  # now disabled to allow for genre zero vectors
    vector = np.zeros((100, 1), dtype=np.float32)
    for genre in genres:
        if 0 <= genre < 100:
            vector[genre, 0] = 1.0
    return vector


def load_image(filepath: str):
    with Image.open(filepath) as image:
        data = np.asarray(image, dtype=np.float32)
        data = data.reshape(-1, 1) / 255.0
        return data






