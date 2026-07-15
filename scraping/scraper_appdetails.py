import json
import requests
import os
import time


def get_all_dumps() -> list[str]:
    dir_path = "appids/"
    return ["appids/" + f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def scrape_at(ids: list[int]):
    if not ids:
        return
    
    base_url = "https://store.steampowered.com/api/appdetails/"
    details = {}
    
    n = len(ids)
    i = 0
    retries = 0
    while i < n:
        time.sleep(1.5)
        curr_id = ids[i]
        full_url = f"{base_url}?appids={curr_id}&l=english&filters=basic,categories,genres,release_date"
        print(f"GET ?appids={curr_id}")
        r = requests.get(full_url)
        if r.ok:
            data = r.json()[str(curr_id)]
            if not data["success"]:
                print(f"UNSUCCESSFUL curr_id={curr_id} status={r.status_code}")
                retries += 1
                if retries > 3:
                    i += 1
                    print("SKIPPED")
                    continue
                print(f"RETRYING {retries} / 3")
                continue
            if data["data"]["type"] != "game":
                print(f"NOT GAME @curr_id={curr_id}  --  SKIPPED")
                i += 1
                continue
            entry = {
                "name": data["data"]["name"],
                "header_image": data["data"]["header_image"],
                "categories": data["data"]["categories"] if "categories" in data["data"] else [],
                "genres": data["data"]["genres"] if "genres" in data["data"] else [],
                "release_date": data["data"]["release_date"]
            }
            details[str(curr_id)] = entry
            retries = 0
            i += 1
        else:
            print(r)
            if r.status_code == 429:
                retries += 1
                extra_delay = max(60, min(10 * retries, 20))
                print(f"429: TOO MANY REQUESTS  --  waiting {extra_delay} extra seconds")
                time.sleep(extra_delay)
            continue

    
    print(f"NUM ENTRIES IN PACKET : {len(details)} / {n}")
    dump_file = f"appdetails/appdetails{ids[0]}.json"
    with open(dump_file, 'x') as f:
        json.dump(details, f, indent=4)
    print(f"DUMPED {dump_file}")
    return


def scrape_appdetails(id_dumps: list[str] = None, limit: int = 999999):
    if not id_dumps:
        id_dumps = get_all_dumps()

    index = 0
    while index < len(id_dumps) and index < limit:
        curr = id_dumps[index]
        print(f"BATCH {curr}")

        with open(curr, 'r') as f:
            raw_ids = json.load(f)
            ids = [x["appid"] for x in raw_ids["response"]["apps"]]
        
        scrape_at(ids)
        index += 1
    return
    

if __name__ == "__main__":
    initial_time = time.time()
    print(f"STARTING AT {time.asctime()}")
    scrape_appdetails()
    print(f"FINISHING AT {time.asctime()}  --  TAKEN {int(time.time() - initial_time)} seconds")







