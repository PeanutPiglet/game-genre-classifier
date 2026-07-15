import json
import requests
import time


START_APPID = 0


def scrape_at(key: str, start_id: int = 0) -> int:
    base_url = "https://partner.steam-api.com/IStoreService/GetAppList/v1/"

    input_json = '{"max_results":50000}'
    if start_id > 0:
        input_json = '{"max_results":50000,"last_appid":' + str(start_id) + '}'

    full_url = f"{base_url}?key={key}&input_json={input_json}"
    print(f"GET {full_url}")
    r = requests.get(full_url)
    print(f"RESPONSE {r}")
    data = r.json()

    with open(f"appids{start_id}.json", 'x') as f:
        json.dump(data, f, indent=4)
    print(f"DUMPED appids{start_id}.json")

    if not "have_more_results" in data["response"]:
        return -1
    if data["response"]["have_more_results"]:
        return data["response"]["last_appid"]
    return -1


def scrape_appids():
    secrets = {}
    with open("../secrets.json") as f:
        secrets = json.load(f)
    publisher_key = secrets["steam-publisher-key"]

    last_id = START_APPID
    while last_id >= 0:
        time.sleep(1.6)
        last_id = scrape_at(publisher_key, last_id)
    return


if __name__ == "__main__":
    scrape_appids()



