import json
import requests
import os


def get_all_dumps() -> list[str]:
    dir_path = "appids/"
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def scrape_at(ids: list[int]):
    if not ids:
        return
    
    base_url = "https://store.steampowered.com/api/appdetails/"
    details = {}
    
    for curr_id in ids:
        full_url = f"{base_url}?appids={curr_id}&l=english&filters=basic,categories,genres,release_date"
        print(f"GET ?appids={curr_id}")
        r = requests.get(full_url)
        if r.ok:
            data = r.json()[str(curr_id)]
            if not data["success"]:
                print(f"UNSUCCESSFUL curr_id={curr_id}")
                continue
            if data["data"]["type"] != "game":
                print(f"NOT GAME @curr_id={curr_id}  --  SKIPPED")
                continue
            entry = {
                "name": data["data"]["name"],
                "header_image": data["data"]["header_image"],
                "categories": data["data"]["categories"],
                "genres": data["data"]["genres"],
                "release_date": data["data"]["release_date"]
            }
            details[str(curr_id)] = entry
        else:
            print(r)
    
    print(f"NUM ENTRIES IN PACKET : {len(details)} / {len(ids)}")
    dump_file = f"appdetails/appdetails{ids[0]}"
    with open(dump_file, 'x') as f:
        json.dump(details, f)
    print(f"DUMPED {dump_file}")
    return


def scrape_appdetails(id_dumps: list[str] = None, batch_limit: int = 999999):
    if not id_dumps:
        id_dumps = get_all_dumps()

    index = 0
    while index < len(id_dumps) and index < batch_limit:
        curr = id_dumps[index]
        print(f"BATCH {curr}")

        with open(curr, 'r') as f:
            raw_ids = json.load(f)
            ids = [x["appid"] for x in raw_ids["response"]["apps"]]
        
        scrape_at(ids)
        index += 1
    return
    







