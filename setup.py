import os
from setuptools import setup

HERE = os.path.dirname(os.path.realpath(__file__))

with open("README.md", "r") as f:
    long_description = f.read()

about = {}
with open(os.path.join(HERE, "asl_ml_camera", "__version__.py"), "r") as f:
    exec(f.read(), about)  # pylint: disable=exec-used

setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    package_dir={"": "."},
    packages=["asl_ml_camera", "asl_ml_camera.tasks"],
    install_requires=[
        "mediapipe",
        "ffmpeg",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "imblearn",
        "pandas",
    ],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["asl_ml_camera = asl_ml_camera.__main__:main"]},
    license=about["__license__"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
)
