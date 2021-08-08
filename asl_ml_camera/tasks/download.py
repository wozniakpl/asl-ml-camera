import shutil
import os
from asl_ml_camera.exit_codes import SUCCESS, FAIL
from asl_ml_camera.utils import error_print


class DownloadTask:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir

    def run(self):
        print("Downloading...")
        try:
            repo_url = "https://github.com/wozniakpl/asl-movies"
            repo_dir = os.path.join(self.artifacts_dir, "asl-movies")
            movies_path = os.path.join(repo_dir, "movies")
            movies_in_artifacts_path = os.path.join(self.artifacts_dir, "movies")
            if os.path.exists(movies_in_artifacts_path):
                print(f"Movies already downloaded in: {movies_in_artifacts_path}")
                return SUCCESS
            os.system(f"git clone {repo_url} {repo_dir}")
            os.rename(movies_path, movies_in_artifacts_path)
            shutil.rmtree(repo_dir)

        except Exception as e:
            error_print(e)
            return FAIL

        return SUCCESS
