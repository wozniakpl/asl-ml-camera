import os
from asl_ml_camera.tasks.camera import convert_predictions, get_winners

HERE = os.path.dirname(os.path.abspath(__file__))


def test_processing_data():
    predictions = [0.0] * 24 + [0.9, 0.1]
    converted = convert_predictions(predictions)
    assert converted["A"] == 0.0
    assert converted["Y"] == 0.9
    assert converted["Z"] == 0.1


def test_getting_winner():
    predictions = [0.0] * 23 + [0.5, 0.4, 0.1]
    converted = convert_predictions(predictions)
    winners = get_winners(converted)
    assert winners[0] == ("X", 0.5)
    assert winners[1] == ("Y", 0.4)
    assert winners[2] == ("Z", 0.1)
