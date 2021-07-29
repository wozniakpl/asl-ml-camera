from asl_ml_camera.subprocess import run_subprocess
import os
from asl_ml_camera.utils import error_print
from asl_ml_camera.exit_codes import SUCCESS, FAIL


def get_all_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


class CuttingTask:
    def __init__(self, artifacts_dir, rate):
        self.rate = rate
        self.artifacts_dir = artifacts_dir

    def cut_movie(self, movie):
        def get_filename(path):
            return os.path.splitext(os.path.basename(path))[0]

        def get_name_of_parent_directory(path):
            return os.path.basename(os.path.dirname(path))

        print("Cutting movie", movie)

        movie_filename = get_filename(movie)
        dirname = get_name_of_parent_directory(movie)

        movie_artifacts_path = os.path.join(self.artifacts_dir, "frames", dirname)
        if not os.path.exists(movie_artifacts_path):
            os.mkdir(movie_artifacts_path)

        subvideo_artifacts_path = os.path.join(
            movie_artifacts_path, movie_filename.replace(" ", "_")
        )
        if not os.path.exists(subvideo_artifacts_path):
            os.mkdir(subvideo_artifacts_path)

        path = subvideo_artifacts_path + "/%4d.jpeg"

        print("Saving as", path)

        invocation = [
            "ffmpeg",
            "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            movie,
            "-r",
            self.rate,
            "-f",
            "image2",
            path,
        ]
        return run_subprocess(invocation)

    def run(self):
        try:
            movies_in_artifacts = [
                file for file in get_all_files(self.artifacts_dir) if ".mov" in file
            ]
            if not os.path.exists(self.artifacts_dir + "/frames"):
                os.mkdir(self.artifacts_dir + "/frames")

            for movie in movies_in_artifacts:
                self.cut_movie(movie)
        except Exception as exception:
            error_print(exception)
            return FAIL
        return SUCCESS
