from PIL import Image
import numpy as np
import os
import time
import json
from common_types import *


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
    batches = get_batches()
    n = len(batches)
    i = 0
    while i < n:
        batch = batches[i]
        manifest = get_manifest(batch)
        n_entries = len(manifest)
        print(f"{n_entries} entries")
        j = 0
        for appid in manifest:
            image = load_image(os.path.join(batch, f"{appid}.jpg"))
            res = network.feedforward(image)
            actual = transform_genres_to_vector(manifest[appid])
            costs = 2 * (res - actual)
            network.backprop(costs)
            network.sgd()

            j += 1
            print(f"{j} / {n_entries}")
        i += 1
        print(f"batch {i} / {n}")
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
    assert len(genres) > 0
    i = 0
    j = 0
    vector = []
    while i < 100:
        if i == genres[j]:
            vector.append([1])
            i += 1
            j += 1
            if j >= len(genres):
                break
            continue
        vector.append([0])
        i += 1

    while i < 100:
        vector.append([0])
        i += 1

    return np.asarray(vector)


def load_image(filepath: str):
    image = Image.open(filepath)
    data = np.asarray(image, dtype=float)
    data = data.flatten()
    data /= 255
    data = data.reshape(-1, 1)
    return data






