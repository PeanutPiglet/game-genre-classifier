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
    def __init__(self, n: int, acc_mean: float, acc_std: float, cost_mean: float, cost_std: float, wrong_size: int):
        self.n = n
        self.acc_mean = acc_mean
        self.acc_std = acc_std
        self.cost_mean = cost_mean
        self.cost_std = cost_std
        self.wrong_size = wrong_size
    def __str__(self):
        return f"acc: {self.acc_mean:.4f}, {self.acc_std:.4f} | costs: {self.cost_mean:.4f}, {self.cost_std:.4f} | size: {self.n} | skipped: {self.wrong_size}"


def test(network: Network, log_every_batch: bool = True, threshold: float = 0.5) -> TestResult:
    begin_time = time.time()
    print(f"starting testing at {time.asctime()}")

    results: list[TestResult] = []
    batches = dataloader.get_batches(directory="test/")
    for batch_path in batches:
        res = test_batch(network=network, batch_path=batch_path, threshold=threshold)
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
        missed
    )
    print(collected)
    return collected


def test_batch(network: Network, batch_path: str, threshold: float = 0.5) -> TestResult:
    manifest = dataloader.get_manifest(batch_path)
    expected_image_size = TESTING_IMAGE_W * TESTING_IMAGE_H * 3
    wrong_size_counter = 0

    n = 0
    acc_list = [0] * len(manifest)
    cost_list = [0] * len(manifest)

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
            acc = 1 - ( np.sum(abs(target - binary)) / len(target) )
            acc_list[n] = acc
            cost = (2 * abs(target - output)).sum() / len(target)
            cost_list[n] = cost
            n += 1
        else:
            wrong_size_counter += 1

    accs = np.fromiter(acc_list[:n], dtype=float)
    costs = np.fromiter(cost_list[:n], dtype=float)

    return TestResult(n, accs.mean(), accs.std(), costs.mean(), costs.std(), wrong_size_counter)
    




