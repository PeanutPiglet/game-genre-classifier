from PIL import Image
import os
import json
import numpy as np


def get_manifest(batch: str) -> dict:
    with open(os.path.join(batch, 'appdetails.json'), 'r') as f:
        data = json.load(f)
        manifest = {appid: data[appid]['genres'] for appid in data}
        return manifest


def get_batches(directory: str = "data/") -> list[str]:
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

    