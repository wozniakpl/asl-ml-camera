import sys
import os
import argparse
from asl_ml_camera.tasks.cutting import CuttingTask
from asl_ml_camera.tasks.download import DownloadTask
from asl_ml_camera.tasks.mediapipe import MediapipeTask
from asl_ml_camera.tasks.dataset import DatasetTask
from asl_ml_camera.tasks.training import TrainingTask
from asl_ml_camera.tasks.camera import CameraTask
from asl_ml_camera.exit_codes import SUCCESS, FAIL
from asl_ml_camera.utils import error_print


def create_argparser():
    parser = argparse.ArgumentParser(
        description="A machine learning taught camera app"
        " that translates your hand to the ASL letter.",
        prog="asl-ml-camera",
        add_help=True,
    )

    modes_group = parser.add_argument_group("modes")
    modes_group.add_argument(
        "-u",
        "--cut",
        dest="cut",
        action="store_true",
        help="path to movie that will be cut to frames",
    )
    modes_group.add_argument(
        "-o",
        "--download",
        dest="download",
        action="store_true",
        help="use, if you want to downlaod training data from google drive",
    )

    cutting_group = parser.add_argument_group(
        title="cut",
        description="Cutting setup",
    )
    cutting_group.add_argument(
        "-r",
        "--frame-rate",
        dest="frame_rate",
        type=str,
        help="frame rate (Hz value, fraction or abbreviation) - "
        "goes straight to ffmpeg under -r",
    )

    modes_group.add_argument(
        "-m",
        "--mediapipe",
        dest="mediapipe",
        action="store_true",
        help="process frames/ directory to find hand landmarks",
    )

    modes_group.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        action="store_true",
        help="merge all .json outputs from --mediapipe to single .csv",
    )

    modes_group.add_argument(
        "-t",
        "--train",
        dest="train",
        action="store_true",
        help="train classifiers and dump them to reuse later",
    )

    modes_group.add_argument(
        "-c",
        "--camera",
        dest="camera",
        action="store_true",
        help="use classifiers with camera to recognize ASL letters",
    )

    parser.add_argument(
        "-a", "--artifacts", dest="artifacts", type=str, help="artifacts directory"
    )

    return parser


def get_args():
    argparser = create_argparser()
    return argparser.parse_args()


def is_config_invalid(args):

    if not args.artifacts:
        error_print("You need to specify artifacts dir.")
        return True

    if args.cut:
        if not args.frame_rate:
            error_print("You need to specify frame rate when using cut mode.")
            return True
    return False


# pylint: disable=too-complex,too-many-return-statements
def run_task(args):
    out = 0
    if args.download:
        out += DownloadTask(args.artifacts).run()
    if args.cut:
        out += CuttingTask(args.artifacts, args.frame_rate).run()
    if args.mediapipe:
        out += MediapipeTask(args.artifacts).run()
    if args.dataset:
        out += DatasetTask(args.artifacts).run()
    if args.train:
        out += TrainingTask(args.artifacts).run()
    if args.camera:
        out += CameraTask(args.artifacts).run()
    if out != 0:
        return FAIL
    return SUCCESS


def main():
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    args = get_args()

    if is_config_invalid(args):
        return FAIL

    if not os.path.exists(args.artifacts):
        os.makedirs(args.artifacts)
    return run_task(args)


if __name__ == "__main__":
    sys.exit(main())
