import sys
import os
import json


def get_existing_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def error_print(msg):
    print(msg, file=sys.stderr)


def get_all_directories(path):
    return [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]


def get_json(path):
    with open(path, "r") as json_data:
        return json.load(json_data)
