import os
import tempfile
import shutil
from contextlib import contextmanager
from tests.utils import run_successfully


HERE = os.path.dirname(os.path.abspath(__file__))


@contextmanager
def temporary_dir_copy(original):
    with tempfile.TemporaryDirectory() as temp_directory:
        original_base = os.path.basename(original)
        temp_path = os.path.join(temp_directory, original_base)
        try:
            shutil.copytree(original, temp_path)
            yield temp_path
        finally:
            shutil.rmtree(temp_path)


def test_no_arguments_should_draw_help(app):
    _ = run_successfully(app)


def test_drawing_help(app):
    _ = run_successfully(app.help())


def test_cutting_videos(app):
    artifacts_dir = os.path.join(HERE, "artifacts", "cutting")

    with temporary_dir_copy(artifacts_dir) as copy:
        assert os.path.exists(os.path.join(copy, "movies", "movie_1", "m-1.mov"))
        assert os.path.exists(os.path.join(copy, "movies", "movie_1", "m-2.mov"))

        _ = run_successfully(app.artifacts(copy).cut().rate("1"))
        os.path.exists(os.path.join(copy, "frames", "movie_1", "m-1", "0001.jpeg"))
        os.path.exists(os.path.join(copy, "frames", "movie_1", "m-1", "0002.jpeg"))
        os.path.exists(os.path.join(copy, "frames", "movie_1", "m-2", "0001.jpeg"))
        os.path.exists(os.path.join(copy, "frames", "movie_1", "m-2", "0002.jpeg"))


def test_getting_mediapipe_info_from_image(app):
    artifacts_dir = os.path.join(HERE, "artifacts", "mediapipe")

    with temporary_dir_copy(artifacts_dir) as copy:
        assert os.path.exists(
            os.path.join(copy, "frames", "movie_1", "m-1", "0001.jpeg")
        )
        assert os.path.exists(os.path.join(copy, "frames", "movie_1", "mapping.json"))

        _ = run_successfully(app.artifacts(copy).mediapipe())

        assert os.path.exists(
            os.path.join(copy, "landmarks", "movie_1", "m-1", "0001.json")
        )
        assert os.path.exists(
            os.path.join(copy, "landmarks", "movie_1", "m-1", "0001.jpeg")
        )


def test_creating_training_set(app):
    artifacts_dir = os.path.join(HERE, "artifacts", "dataset")
    with temporary_dir_copy(artifacts_dir) as copy:
        assert os.path.exists(
            os.path.join(copy, "landmarks", "movie_1", "m-1", "0001.json")
        )
        assert os.path.exists(
            os.path.join(copy, "landmarks", "movie_1", "m-2", "0002.json")
        )
        assert os.path.exists(
            os.path.join(copy, "landmarks", "movie_1", "m-3", "0003.json")
        )

        _ = run_successfully(app.artifacts(copy).dataset())

        dataset_path = os.path.join(copy, "dataset", "full.csv")
        assert os.path.exists(dataset_path)

        with open(dataset_path) as dataset_file:
            first_line = dataset_file.readline().rstrip().split(",")
            assert "ASL_Video" in first_line
            assert "Letter" in first_line
            assert "x0" in first_line
            assert "y0" in first_line
            assert "z0" in first_line
            assert "x20" in first_line
            assert "y20" in first_line
            assert "z20" in first_line


def ensure_directory_does_not_exist(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    assert not os.path.exists(path)


def test_training_classifiers(app):
    artifacts_dir = os.path.join(HERE, "artifacts", "training")
    with temporary_dir_copy(artifacts_dir) as copy:
        assert os.path.exists(os.path.join(copy, "dataset", "full.csv"))
        ensure_directory_does_not_exist(os.path.join(copy, "training"))

        _ = run_successfully(app.artifacts(copy).train())

        assert os.path.exists(os.path.join(copy, "training", "rfc.pkl"))
        assert os.path.exists(os.path.join(copy, "training", "rfc.json"))
        assert os.path.exists(os.path.join(copy, "training", "gnb.pkl"))
        assert os.path.exists(os.path.join(copy, "training", "gnb.json"))
