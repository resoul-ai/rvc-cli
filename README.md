# rvc_cli

rvc_cli is a voice to voice cli package built on rvc

Instructions are follow along. If you download a Travis Baldree book these instructions will work as is. Only things that change are names in `.env` w.r.t. respect whichever voice you are converting to and adding absolute paths if you don't want to use relative.

## Setup
```
uv venv rvc_cli --python 3.10.14
source rvc_cli/bin/activate
uv pip install -e .
```

## Data

### Audible
Use [libation](https://github.com/rmcrackan/Libation) to download unencrypted `.mp3`'s from your library

Example File Tree (inside `rvc-cli`):
```
-data
    |-books
        |-Uncrowned.mp3
    |-clips
```

### Preprocessing
Folder of input data should be full of approximately ~1 minute `.mp3` clips
Use your downloaded audiobook + `python -m rvc_cli create_dataset_from_large_mp3`
> requires ffmpeg installed on machine

Example:
```
python -m rvc_cli create-dataset-from-large-mp3 --input-file-path "
data/books/Uncrowned.mp3" --output-directory "data/clips"
```


## Usage
Fill in you `.env` and `config.yml` files (If following along with examples, default files will work as is)
Replace the samples found inside of `rvc-cli/configuration`

### Download RVC models 

Example
```
python -m rvc_cli download-rvc-models-to --download-directory "rvc-models/"
```

### Train

Example (defaults we're set to align with example walkthrough)
```
python -m rvc_cli train
```

Loss and other metrics can be viewed via tensorboard
Example:
```
tensorboard --logdir rvc-models/results/travis-baldree
```

### Infer

Provide your own audio files to convert and place them in `data/conversion/pre`.
Otherwise, specify input directory and output directory in the cli

```
python -m rvc_cli infer --help
```


# Credit
- [rvc-webui](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/3548b4f1a55336629955c0d51deeb24b6de9c46e/docs/en/README.en.md)
- [rvc](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)



# Developer Notes

in order to fix the rvc backend to be cli compatible with click all argparse instances were replaced. If you want to use anything from those that we're buried you would access them via env variables:

```

export SAVE_EVERY_EPOCH=5
export TOTAL_EPOCH=200
export EXPERIMENT_DIR="my_experiment"
# ... set other variables as needed

export AUDIO_FILE="/path/to/your/audio.wav"
export OUTPUT_DIR="/path/to/output/directory"
export DB_THRESH="-35"
# ... set other variables as needed

export ALGORITHM="min_mag"
export MODEL_PARAMS="/path/to/your/model/params.json"
export OUTPUT_NAME="my_output"
export VOCALS_ONLY="false"
export INPUT_FILES="/path/to/input1.wav,/path/to/input2.wav"

export PORT=8000
export PYCMD="/usr/bin/python3"
export COLAB="true"
export NOPARALLEL="false"
export NOAUTOOPEN="true"
export DML="false"
```

args from files:
- infer/../utils
- infer/../slicer2
- infer/../spec_utils
- configs/config.py

respectively