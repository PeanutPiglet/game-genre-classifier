import time
import os
import numpy as np
from common_types import *
import dataloader


TESTING_IMAGE_W = 128
TESTING_IMAGE_H = 64


class TestResult:
    n: int
    acc_mean: float
    acc_std: float
    cost_mean: float
    cost_std: float
    wrong_size: int
    fp_rate: float
    fn_rate: float
    def __init__(self, n: int, acc_mean: float, acc_std: float, cost_mean: float, cost_std: float, wrong_size: int, fp_rate: float, fn_rate: float):
        self.n = n
        self.acc_mean = acc_mean
        self.acc_std = acc_std
        self.cost_mean = cost_mean
        self.cost_std = cost_std
        self.wrong_size = wrong_size
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate
    def __str__(self):
        return f"acc: {self.acc_mean:.4f}, {self.acc_std:.4f} | fp: {self.fp_rate:.4f}, fn: {self.fn_rate:.4f} | costs: {self.cost_mean:.4f}, {self.cost_std:.4f} | size: {self.n} | skipped: {self.wrong_size}"


def test(network: Network, log_every_batch: bool = True, threshold: float = 0.5, placeholder_output: bool = False) -> TestResult:
    begin_time = time.time()
    print(f"starting testing at {time.asctime()}")

    results: list[TestResult] = []
    batches = dataloader.get_batches(directory="test/")
    for batch_path in batches:
        res = test_batch(network=network, batch_path=batch_path, threshold=threshold, placeholder_output=placeholder_output)
        results.append(res)
        if log_every_batch:
            print(res)

    print(f"ending at {time.asctime()}  --  taken {int(time.time() - begin_time)} seconds")

    total = sum(x.n for x in results)
    missed = sum(x.wrong_size for x in results)

    collected = TestResult(
        total,
        sum(x.n * x.acc_mean for x in results) / total,
        sum(x.n * x.acc_std for x in results) / total,
        sum(x.n * x.cost_mean for x in results) / total,
        sum(x.n * x.cost_std for x in results) / total,
        missed,
        sum(x.n * x.fp_rate for x in results) / total,
        sum(x.n * x.fn_rate for x in results) / total
    )
    print(collected)
    return collected


def test_batch(network: Network, batch_path: str, threshold: float = 0.5, placeholder_output: bool = False) -> TestResult:
    manifest = dataloader.get_manifest(batch_path)
    expected_image_size = TESTING_IMAGE_W * TESTING_IMAGE_H * 3
    wrong_size_counter = 0

    if not placeholder_output:
        genres = dataloader.get_genres("test/genres.json", True)

    n = 0
    acc_list = [0] * len(manifest)
    cost_list = [0] * len(manifest)
    fp_list = [0] * len(manifest)
    fn_list = [0] * len(manifest)

    for appid in manifest:
        image = dataloader.load_image(os.path.join(batch_path, f"{appid}.jpg"))
        if len(image) == expected_image_size:
            target = dataloader.transform_genres_to_vector(manifest[appid])
            output = network.feedforward(image)
            binary = output.copy()
            for entry in binary:
                if entry[0] >= threshold:
                    entry[0] = 1
                else:
                    entry[0] = 0
            if placeholder_output:
                raw_diff = target - binary
                acc = 1 - ( np.sum(abs(raw_diff)) / len(target) )
                acc_list[n] = acc
                cost = (2 * abs(target - output)).sum() / len(target)
                cost_list[n] = cost
                fp_list[n] = np.sum(np.clip(raw_diff, -1, 0)) * -1 / len(target)
                fn_list[n] = np.sum(np.clip(raw_diff, 0, 1)) / len(target)
            else:
                raw_diff = target - binary
                acc_diff = abs(raw_diff)
                acc = 1 - ( sum(acc_diff[i, 0] for i in genres) / len(genres) )
                acc_list[n] = acc
                cost_diff = 2 * abs(target - output)
                cost = sum(cost_diff[i, 0] for i in genres) / len(genres)
                cost_list[n] = cost
                fp_list[n] = sum(min(raw_diff[i, 0], 0) for i in genres) * -1 / len(genres)
                fn_list[n] = sum(max(raw_diff[i, 0], 0) for i in genres) / len(genres)
            n += 1
        else:
            wrong_size_counter += 1

    accs = np.fromiter(acc_list[:n], dtype=float)
    costs = np.fromiter(cost_list[:n], dtype=float)
    fps = np.fromiter(fp_list[:n], dtype=float)
    fns = np.fromiter(fn_list[:n], dtype=float)

    return TestResult(n, accs.mean(), accs.std(), costs.mean(), costs.std(), wrong_size_counter, fps.mean(), fns.mean())
    




