import os
import cv2
import mediapipe as mp
from joblib import load
from asl_ml_camera.exit_codes import SUCCESS

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
WHITE = [255, 255, 255]
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color_red = (0, 0, 255)  # BGR
# color_blue = (255, 0, 0) # BGR
thickness = 2


def convert_predictions(predictions):
    predictions_map = {}
    for i in range(0, 26):
        predictions_map[chr(i + 65)] = predictions[i]
    return predictions_map


def get_winners(predictions):
    return [
        (item[0], predictions[item[0]])
        for item in sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    ]


def draw_predictions(image, predictions):
    def write(img, x, y, txt):
        print(f"writing {txt} at ({x},{y})")
        return cv2.putText(
            img,
            txt,
            (int(x), int(y)),
            font,
            fontScale,
            color_red,
            thickness,
            cv2.LINE_AA,
        )

    (y, x, _) = image.shape  # (480,640,3)

    winners = get_winners(predictions)
    x_distance = x / 5
    left_margin = 5
    bottom_margin = 5
    image = write(image, left_margin, y - bottom_margin, "Predictions:")

    def write_winner(img, index):
        img = write(
            img,
            left_margin + x / 3 + index * x_distance,
            y - bottom_margin,
            winners[index][0],
        )
        return write(
            img,
            left_margin + x / 3 + index * x_distance + 25,
            y - bottom_margin,
            f":{winners[index][1]:.2}",
        )

    image = write_winner(image, 0)
    image = write_winner(image, 1)
    image = write_winner(image, 2)
    return image


def convert_to_array(landmarks):
    output = []
    for data in landmarks.landmark:
        output.extend([data.x, data.y, data.z])
    return output


def get_predictions(classifier, landmarks):
    marks = convert_to_array(landmarks)
    prediction = classifier.predict_proba([marks])
    return prediction


def draw_ml_info(image, predictions):
    print("drawing predictions", predictions)
    predictions_map = convert_predictions(predictions[0])
    image = draw_predictions(image, predictions_map)
    return image


class CameraTask:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir

    def get_classifier(self):
        return load(os.path.join(self.artifacts_dir, "training", "rfc.pkl"))

    def run(self):
        print("Camera...")
        classifier = self.get_classifier()
        hands = mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hands_landmarks = list(results.multi_hand_landmarks)
                if len(hands_landmarks) == 1:  # only one for now
                    landmarks = hands_landmarks[0]
                    mp_drawing.draw_landmarks(
                        image, landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    predictions = get_predictions(classifier, landmarks)
                    image = draw_ml_info(image, predictions)
            cv2.imshow("MediaPipe Hands", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        hands.close()
        cap.release()

        return SUCCESS
