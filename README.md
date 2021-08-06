# ASL Camera

This project uses the camera to translate your hand to ASL letters. Below you can find out, how the ML model was trained.

# How to use

Clone the repo and call

```
pip3 install -e . # to install and call like: asl_ml_camera <args>
# or install the required packages and call directly:
pip3 install -r requirements.txt
python3 -m asl_ml_camera
```

Help should be printed.

Then, download the [training data](https://drive.google.com/drive/folders/1fYZ_ROFbDsHSjQvgSXHGQdRHB8DmzYda?usp=sharing) (I did what I could to integrate this with some Google Drive API but it's just not worth it) and put it in the `artifacts` directory (or however you name it), so it looks like this:

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
