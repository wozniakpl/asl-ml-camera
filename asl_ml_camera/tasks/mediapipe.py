import mediapipe as mp
import cv2
import os
import json
from asl_ml_camera.utils import get_existing_path, get_all_directories, get_json
from asl_ml_camera.exit_codes import SUCCESS

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7
)


def convert_to_pretty(landmarks):
    output = []
    for i in range(0, len(landmarks.landmark)):
        data = landmarks.landmark[i]
        output.append({"x": data.x, "y": data.y, "z": data.z})
    return output


def get_mediapipe_info(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        return None, None

    annotated_image = image.copy()

    labels = [info.classification.pop() for info in results.multi_handedness]

    hands_data = {}
    count = 0
    for hand_landmarks in results.multi_hand_landmarks:
        label = labels[count].label
        hands_data[label] = convert_to_pretty(hand_landmarks)
        count += 1

        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

    annotated_image = cv2.flip(annotated_image, 1)
    return annotated_image, hands_data


def get_video_number(path):
    # asserting that video ends like something_0001.jpeg
    # extension should be removed and the text before the number should be returned
    return str(
        int(path.split("/")[-1].split(".")[0].rsplit("-")[-1])
    )  # str(int(_)) trims leading 0s


class MediapipeTask:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir

    def run(self):
        print("Processing...")
        print(self.artifacts_dir)
        movie_directories = get_all_directories(
            os.path.join(self.artifacts_dir, "frames")
        )

        ml_output_path = get_existing_path(
            os.path.join(self.artifacts_dir, "landmarks")
        )

        for movie in movie_directories:
            movie_output_path = get_existing_path(
                os.path.join(ml_output_path, os.path.basename(movie))
            )

            mapping_path = os.path.join(
                self.artifacts_dir, "movies", os.path.basename(movie), "mapping.json"
            )
            mapping = get_json(mapping_path)
            self.analyze_each_frame(movie, movie_output_path, mapping)

        return SUCCESS

    def analyze_each_frame(self, movie, output_path, mapping):
        def get_all_frames(path):
            return [f for f in os.listdir(path) if f.endswith(".jpeg")]

        frames_directories = get_all_directories(movie)
        for frames_directory in frames_directories:
            frames = get_all_frames(frames_directory)
            output_frames_path = get_existing_path(
                os.path.join(output_path, os.path.basename(frames_directory))
            )
            video_number = get_video_number(frames_directory)
            letter = mapping[video_number]
            for frame in frames:
                frame_path = os.path.join(frames_directory, frame)
                print("Processing", frame_path)

                original_image = cv2.flip(cv2.imread(frame_path), 1)
                image, landmarks = get_mediapipe_info(original_image)

                if landmarks == {} or image is None:
                    continue

                jpeg_filepath = os.path.join(output_frames_path, frame)
                print("Saving image", jpeg_filepath)
                cv2.imwrite(jpeg_filepath, image)

                json_filepath = jpeg_filepath.replace(".jpeg", ".json")
                with open(f"{json_filepath}", "w") as outfile:
                    data = {"landmarks": landmarks, "letter": letter}
                    json.dump(data, outfile, indent=2)
                    print("Saving data", outfile)
