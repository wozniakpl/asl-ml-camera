import os
import csv
from itertools import chain
from asl_ml_camera.utils import get_all_directories, get_json
from asl_ml_camera.exit_codes import SUCCESS


def flatten(container):
    return list(chain.from_iterable(container))


def get_landmarks(data):
    landmarks = data["landmarks"]
    if "Right" in landmarks and "Left" in landmarks:
        return None

    if "Right" in landmarks:
        right = landmarks["Right"]
        return flatten([[a["x"], a["y"], a["z"]] for a in right])
    if "Left" in landmarks:
        left = landmarks["Left"]
        return flatten([[a["x"], a["y"], a["z"]] for a in left])

    return None


def get_json_files(path):
    return [
        os.path.join(dirpath, file)
        for dirpath, _, files in os.walk(path)
        for file in files
        if file.endswith(".json")
    ]


class DatasetTask:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir

    def get_data_from_jsons(self, movie_directories):
        csv_data_container = []
        for movie_directory in movie_directories:
            print("Getting data from", movie_directory)

            files = get_json_files(movie_directory)
            for file_path in files:
                data = get_json(file_path)

                landmarks = get_landmarks(data)
                if landmarks is None:
                    continue

                letter = data["letter"]
                letter_code = ord(letter) - 65

                asl_video = os.path.basename(movie_directory)
                video = os.path.dirname(file_path)
                frame = file_path.split("/")[-1].split(".")[0]

                csv_data_container.append(
                    flatten(
                        [
                            [asl_video],
                            [video],
                            [frame],
                            landmarks,
                            [letter_code],
                        ]
                    )
                )
        return csv_data_container

    def run(self):
        print("Generating datasets")
        movie_directories = get_all_directories(
            os.path.join(self.artifacts_dir, "landmarks")
        )

        dataset_dir = os.path.join(self.artifacts_dir, "dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        csv_data_container = self.get_data_from_jsons(movie_directories)

        with open(os.path.join(dataset_dir, "full.csv"), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            labels = [f"{v}{i}" for i in range(0, 21) for v in ["x", "y", "z"]]
            row = ["ASL_Video", "Subvideo", "Frame"] + labels + ["Letter"]
            writer.writerow(row)
            for line in csv_data_container:
                writer.writerow(line)

        return SUCCESS
