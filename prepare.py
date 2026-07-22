import os
import json
import shutil


BATCH_SIZE = 1000


def prepare():
    if not verify_folder_ready():
        return
    
    print("POPULATING APPDETAILS")
    populate_appdetails()
    print("COPYING HEADER IMAGES")
    populate_headers()

    return


def populate_appdetails():
    curr_batch = {}
    first_id = 0
    appdetails = get_appdetails()
    n = len(appdetails)
    i = 0

    for details_file in appdetails:
        with open(details_file, 'r') as f:
            data = json.load(f)
            for appid in data:
                if not isinstance(data[appid]["header_image"], str):
                    print(f"WARNING: caught entry ({appid}) without valid header image url  --  skipped")
                    continue
                if len(curr_batch) == 0:
                    first_id = appid
                curr_batch[appid] = {
                    "genres": [int(entry["id"]) for entry in data[appid]["genres"]]
                }
                if len(curr_batch) >= BATCH_SIZE:
                    save_batch(curr_batch, first_id)
                    curr_batch = {}

        i += 1
        print(f"{i} / {n}")
    
    if len(curr_batch) > 0:
        save_batch(curr_batch, first_id)
    return


def save_batch(batch: dict, suffix: int):
    os.makedirs(f"data/batch{suffix}")
    with open(f"data/batch{suffix}/appdetails.json", 'x') as f:
        json.dump(batch, f)
    return


def populate_headers():
    partition = get_scraped_header_partition()
    if len(partition) == 0:
        print("ERROR: no scraped headers obtained")
        return
    
    partition.append(999999999999)
    curr_i = 0
    curr_val = partition[curr_i]
    threshold = partition[curr_i + 1]

    batches = get_batches()
    n = len(batches)
    i = 0

    for batch_path in batches:
        with open(f"{batch_path}/appdetails.json", 'r') as f:
            data = json.load(f)
            for appid in data:
                inted = int(appid)
                if inted >= threshold:
                    while inted >= threshold:
                        curr_i += 1
                        threshold = partition[curr_i + 1]
                    curr_val = partition[curr_i]
                shutil.copy(
                    f"scraping/headers/headers{curr_val}/header{appid}.jpg",
                    f"{batch_path}/{appid}.jpg"
                )
        i += 1
        print(f"{i} / {n}")
    return


def verify_folder_ready():
    for p in ["scraping/appdetails", "scraping/headers"]:
        if not os.path.exists("scraping") or not os.path.isdir("scraping"):
            print(f"ERROR: ./{p} folder not found")
            return False

    if not os.path.exists("data") or not os.path.isdir("data"):
        print("ERROR: ./data folder not found")
        return False
    with os.scandir("data") as folder:
        if any(folder):
            print("ERROR: ./data folder is not empty  --  please remove its content")
            return False
    return True


def get_appdetails() -> list[str]:
    dir_path = "scraping/appdetails/"
    listed = [dir_path + f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    listed.sort(key=lambda s: int(s.removeprefix(dir_path + "appdetails").removesuffix(".json")))
    return listed


def get_batches() -> list[str]:
    dir_path = "data/"
    listed = [dir_path + f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    listed.sort(key=lambda s: int(s.removeprefix(dir_path + "batch")))
    return listed


def get_scraped_header_partition() -> list[int]:
    dir_path = "scraping/headers/"
    partition = [int(f.removeprefix("headers"))
                 for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    partition.sort()
    return partition


if __name__ == "__main__":
    prepare()





