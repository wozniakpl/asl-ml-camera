from asl_ml_camera.exit_codes import SUCCESS


class DownloadTask:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir

    def run(self):
        print("Downloading...")
        return SUCCESS
