import os
import json
from asl_ml_camera.exit_codes import SUCCESS
from asl_ml_camera.utils import get_existing_path
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from joblib import dump

RANDOM_STATE = 42


class Classifier:
    def __init__(self, name, model):
        self.name = name
        self.model = model


class TrainingTask:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir
        self.training_output_dir = get_existing_path(
            os.path.join(self.artifacts_dir, "training")
        )

    def feed(
        self,
        classifier,
        samples_train,
        samples_test,
        sample_class_train,
        sample_class_test,
    ):

        trained = classifier.fit(samples_train, sample_class_train.values.ravel())

        predictions = trained.predict(samples_test)
        clf_report = classification_report(
            sample_class_test, predictions, output_dict=True, zero_division=0
        )
        return clf_report, trained

    def train(self, dataset, classifier):
        samples_orig = dataset.transpose()[:-1].transpose()
        sample_class_orig = dataset.transpose()[-1:].transpose()

        samples, sample_class = RandomUnderSampler(
            random_state=RANDOM_STATE
        ).fit_resample(samples_orig, sample_class_orig)

        X_train, X_test, y_train, y_test = train_test_split(
            samples, sample_class, random_state=RANDOM_STATE, test_size=0.25
        )

        report, trained_classifier = self.feed(
            classifier.model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        classifier_name = classifier.name.lower()
        dump(
            trained_classifier,
            os.path.join(self.training_output_dir, f"{classifier_name}.pkl"),
        )

        with open(f"{self.training_output_dir}/{classifier_name}.json", "w") as outfile:
            json.dump(report, outfile, indent=2)

    def run(self):
        dataset_dir = os.path.join(self.artifacts_dir, "dataset")
        dataset_path = os.path.join(dataset_dir, "full.csv")
        original_dataset = pd.read_csv(dataset_path)

        trimmed_dataset = original_dataset.drop(
            labels=["ASL_Video", "Subvideo", "Frame"], axis=1
        )

        classifiers = [
            Classifier("RFC", RandomForestClassifier(random_state=RANDOM_STATE)),
            Classifier("GNB", GaussianNB()),
        ]
        print("Training...")
        print(trimmed_dataset)
        for classifier in classifiers:
            self.train(trimmed_dataset, classifier)

        return SUCCESS
