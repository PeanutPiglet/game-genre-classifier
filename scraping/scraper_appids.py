import json
import requests
import time


DEFAULT_START_APPID = 0
DEFAULT_BATCH_SIZE = 20
DEFAULT_BATCH_LIMIT = 20


def scrape_at(key: str, start_id: int, batch_size: int) -> int:
    base_url = "https://partner.steam-api.com/IStoreService/GetAppList/v1/"

    input_json = f'{{"max_results":{batch_size}}}'
    if start_id > 0:
        input_json = f'{{"max_results":{batch_size},"last_appid":{start_id}}}'

    full_url = f"{base_url}?key={key}&input_json={input_json}"
    print(f"GET {full_url}")
    r = requests.get(full_url)
    print(f"RESPONSE {r}")
    data = r.json()

    with open(f"appids/appids{start_id}.json", 'x') as f:
        json.dump(data, f, indent=4)
    print(f"DUMPED appids{start_id}.json")

    if not "have_more_results" in data["response"]:
        return -1
    if data["response"]["have_more_results"]:
        return data["response"]["last_appid"]
    return -1


def scrape_appids(start_id: int = DEFAULT_START_APPID,
                  batch_size = DEFAULT_BATCH_SIZE, batch_limit = DEFAULT_BATCH_LIMIT) -> list[str]:
    secrets = {}
    with open("../secrets.json") as f:
        secrets = json.load(f)
    publisher_key = secrets["steam-publisher-key"]

    last_id = start_id
    dumps = []
    count = 0
    while last_id >= 0 and count < batch_limit:
        time.sleep(1.6)
        count += 1
        dumps.append(f"appids/appids{last_id}.json")
        last_id = scrape_at(publisher_key, last_id, batch_size)

    return dumps


if __name__ == "__main__":
    scrape_appids()



