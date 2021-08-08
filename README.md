# ASL Camera

This project uses the camera to translate your hand to ASL letters. Below you can find out, how the ML model was trained.

# How to use

Install from the main branch:

```
pip3 install git+https://github.com/wozniakpl/asl-ml-camera.git
```

or clone the repo and call:

```
pip3 install -e . # to install and call like: asl_ml_camera <args>
# or install the required packages and call directly:
pip3 install -r requirements.txt
python3 -m asl_ml_camera <args>
```

With no args, help should be printed. You need `git` to get the input for ML on the `--download` step. It downloads the movies from another repository, so this one wouldn't be that large. It's available [here](https://github.com/wozniakpl/asl-movies). Also, you need to have `ffmpeg` installed. It's used in the `--cut` step, when movie is split to frames.

```
artifacts/
└── movies
    └── movie_1
        ├── mapping.json
        ├── <something>-0001.mov
        ├── <something>-0002.mov
        ...
    └── movie_2
    ...
```

This directory will be used to store all the stages of the data that will be processed.
`mapping.json` should look like this:

```
{
    "1": "A",
    "2": "B",
    "3": "C",
    ...
}
```

so the names of the movies **MUST** end with data convertible to number between `-` and `.extension`.
To cut the movies to .jpeg frames (cutting 5 frames per 1 second of the movie), call:

```
asl_ml_camera -a artifacts --cut --frame-rate 5
```

Make sure to have `ffmpeg` installed.

Then, you extract the hand landmarks with mediapipe. It uses data from `artifacts/frames` and `artifacts/**/mapping.json` to generate .json files in `artifacts/landmarks`.

```
asl_ml_camera -a artifacts --mediapipe
```

To generate dataset from the .json files and save it to .csv, use:

```
asl_ml_camera -a artifacts --dataset
```

and check `artifacts/dataset/`.

Next step is to train the system. This will populate `artifacts/training` with trained classifiers and reports about their statistical params.

```
asl_ml_camera -a artifacts --train
```

To see the trained classifiers in action, prepare your USB camera and call:

```
asl_ml_camera -a artifacts --camera
```

Close with ESC.

You can call everything at once like this:

```
asl_ml_camera -a artifacts --download --cut --frame-rate 5 --mediapipe --dataset --train --camera
```

The invocation above took around 4 minutes on my PC, when having only movie_1 in artifacts.

# Development

To run all tests, lints etc. use: `tox`. To see what it does, check out `tox.ini`.
