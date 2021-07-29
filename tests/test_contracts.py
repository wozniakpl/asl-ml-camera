from tests.utils import run_unsuccessfully


def test_having_to_specify_the_artifacts_directory(app):
    _ = run_unsuccessfully(app.cut())


def test_having_to_specify_the_frame_rate_when_cutting(app):
    _ = run_unsuccessfully(app.artifacts("artifacts").cut())
