import sys
import pytest
from asl_ml_camera.subprocess import run_subprocess


class App:
    def __init__(self):
        self.arguments = []

    def invoke(self):
        invocation = [sys.executable, "-m", "asl_ml_camera", *self.arguments]
        print(f'[DBG] invocation: {" ".join([str(i) for i in invocation])}')
        self.arguments = []
        result = run_subprocess(invocation)
        print("[DBG] Finished process")
        print(result)
        return result

    def help(self):
        self.arguments.append("--help")
        return self

    def cut(self):
        self.arguments.append("--cut")
        return self

    def rate(self, rate):
        self.arguments.extend(["--frame-rate", rate])
        return self

    def artifacts(self, path):
        self.arguments.extend(["--artifacts", path])
        return self

    def download(self):
        self.arguments.append("--download")
        return self

    def mediapipe(self):
        self.arguments.append("--mediapipe")
        return self

    def dataset(self):
        self.arguments.append("--dataset")
        return self

    def train(self):
        self.arguments.append("--train")
        return self


@pytest.fixture
def app():
    return App()
