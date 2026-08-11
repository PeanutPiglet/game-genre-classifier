import json
import requests
import os
import time


def get_all_appdetails() -> list[str]:
    dir_path = "appdetails/"
    return ["appdetails/" + f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def scrape_at(entries, records: dict[str, str]):
    if len(entries) == 0:
        print("WARNING: RECEIVED EMPTY entries  --  skipped")
        return
    initial_len_records = len(records)

    first = ""
    for appid in entries:
        first = appid
        break
    if not first:
        print(f"INVALID first : {first}")
        return
    os.makedirs(f"headers/headers{first}")

    for appid in entries:
        time.sleep(0.2)
        print(f"GET @{appid}")
        header_url = entries[appid]["header_image"]
        if not isinstance(header_url, str):
            print("WARNING: non-str header_url  --  skipped")
        r = requests.get(header_url)
        if r.ok:
            out_path = f"headers/headers{first}/header{appid}.jpg"
            with open(out_path, 'xb') as out:
                out.write(r.content)
                records[appid] = out_path
        else:
            print(r)
    print(f"HEADERS SAVED : {len(records) - initial_len_records} / {len(entries)}")
    return


def scrape_headers(appdetails: list[str] = None, limit: int = 999999) -> dict[str, str]:
    if not appdetails:
        appdetails = get_all_appdetails()

    os.makedirs("headers/", exist_ok=True)

    records = {}
    index = 0
    while index < len(appdetails) and index < limit:
        curr = appdetails[index]
        with open(curr, 'r') as f:
            data = json.load(f)
            scrape_at(data, records)
        index += 1

    return records


if __name__ == "__main__":
    initial_time = time.time()
    print(f"STARTING AT {time.asctime()}")
    records = scrape_headers()
    print(f"FINISHING AT {time.asctime()}  --  took {int(time.time() - initial_time)} seconds")
    print("DUMPING RECORDS INTO header_links.json")
    with open("header_links.json", 'w') as f:
        json.dump(records, f)



