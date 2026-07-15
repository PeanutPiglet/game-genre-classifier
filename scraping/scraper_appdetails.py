import json
import requests
import os


def get_all_dumps() -> list[str]:
    dir_path = "appids/"
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def scrape_appdetails(id_dumps: list[str] = None):
    if not id_dumps:
        id_dumps = get_all_dumps()

    
    







