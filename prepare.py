import os
import json
import shutil


BATCH_SIZE = 1000


def prepare():
    if not verify_folder_ready():
        return
    
    curr_batch = {}
    first_id = 0
    appdetails = get_appdetails()
    n = len(appdetails)
    i = 0

    for details_file in appdetails:
        with open(details_file, 'r') as f:
            data = json.load(f)
            for appid in data:
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


def get_appdetails():
    dir_path = "scraping/appdetails/"
    listed = [dir_path + f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    listed.sort()
    return listed


if __name__ == "__main__":
    prepare()





