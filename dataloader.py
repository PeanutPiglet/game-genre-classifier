from PIL import Image
import os
import json
import numpy as np

IMAGE_W = 128
IMAGE_H = 64
IMAGE_C = 3


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


def load_image_2d(filepath: str, resize: bool = False):
    with Image.open(filepath) as image:
        image = image.convert('RGB')
        if resize:
            image = image.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        data = np.asarray(image, dtype=np.float32) / 255.0
        if data.shape[:2] != (IMAGE_H, IMAGE_W):
            raise ValueError(f"Expected image size {IMAGE_W}x{IMAGE_H}, got {data.shape[1]}x{data.shape[0]}")
        return data.transpose(2, 0, 1)


def load_image(filepath: str, resize: bool = False):
    data = load_image_2d(filepath, resize)
    return data.reshape(-1, 1)


def get_genres(filepath: str, int_key: bool = False) -> dict:
    with open(filepath, 'r') as f:
        data = json.load(f)
        if not isinstance(data, dict):
            return {}
        if not int_key:
            return data
        transformed = {int(i): data[i] for i in data}
        return transformed

    