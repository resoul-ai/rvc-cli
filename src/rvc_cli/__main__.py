import logging
import os
import tempfile
from pathlib import Path
from typing import List

import click
import dotenv
import numpy as np
import torch
import yaml
from scipy.io import wavfile

from rvc_cli.api.create_dataset import process_audio_file
from rvc_cli.api.download_models import download_rvc_models
from rvc_cli.api.inference import API_VC
from rvc_cli.configs.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.float32

# Default paths
DEFAULT_ENV_PATH = Path("configuration/.env")
DEFAULT_CONFIG_FILE = Path("configuration/config.yml")
DEFAULT_INPUT_AUDIO_DIR = Path("data/conversion/pre/")
DEFAULT_OUTPUT_DIR = Path("data/conversion/post")


# TODO: move api/
def split_audio(
    sr: int, audio_data: np.ndarray, max_duration_seconds: int = 3600
) -> List[np.ndarray]:
    """Split audio into segments no longer than max_duration_seconds."""
    samples_per_segment = sr * max_duration_seconds
    total_samples = len(audio_data)

    segments = []
    for start in range(0, total_samples, samples_per_segment):
        end = min(start + samples_per_segment, total_samples)
        segment = audio_data[start:end]
        segments.append(segment)

    return segments


def combine_audio_files(temp_dir: Path, output_file: Path) -> None:
    """Combine multiple temporary WAV files into a single output file."""
    # Get all temporary wav files sorted by segment number
    temp_files = sorted(
        [f for f in temp_dir.glob("segment_*.wav")],
        key=lambda x: int(x.stem.split("_")[1]),
    )

    if not temp_files:
        raise ValueError("No temporary files found to combine")

    # Read the first file to get sample rate and initialize combined audio
    sr, first_segment = wavfile.read(str(temp_files[0]))
    combined_audio = [first_segment]

    # Read and append each subsequent file
    for temp_file in temp_files[1:]:
        curr_sr, segment_data = wavfile.read(str(temp_file))
        if curr_sr != sr:
            raise ValueError(f"Inconsistent sample rate in {temp_file}")
        combined_audio.append(segment_data)

    # Concatenate all segments and write to output file
    final_audio = np.concatenate(combined_audio)
    wavfile.write(output_file, sr, final_audio)

    # Clear the combined_audio list to free memory
    combined_audio.clear()


def load_config(file_path: str):
    config_path = os.getenv("CONFIG_PATH", file_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@click.group()
def cli():
    """Voice Conversion CLI for inferring and training voice conversion models."""
    pass


@cli.command()
@click.option(
    "--env-path",
    type=click.Path(exists=True),
    default=DEFAULT_ENV_PATH,
    help="Path to the .env file",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    default=DEFAULT_CONFIG_FILE,
    help="Path to the config YAML file",
)
@click.option(
    "--input-audio-dir",
    type=click.Path(exists=True),
    default=DEFAULT_INPUT_AUDIO_DIR,
    help="Directory containing input audio files",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=DEFAULT_OUTPUT_DIR,
    help="Directory to save output audio files",
)
def infer(env_path, config_file, input_audio_dir, output_dir):
    """Process multiple chapters (contents of provided directory) using parallel TTS inference."""
    env_path = Path(env_path)
    config_file = Path(config_file)
    input_audio_dir = Path(input_audio_dir)
    output_dir = Path(output_dir)

    vc_single_config = load_config(config_file)

    dotenv.load_dotenv(dotenv_path=env_path)
    model_name = os.getenv("model_name")
    config = Config()
    vc = API_VC(config)
    vc.get_vc(sid=model_name)

    os.makedirs(output_dir, exist_ok=True)

    # Create a persistent temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for file in os.listdir(input_audio_dir):
            if file.endswith(".wav"):
                input_file = input_audio_dir / file
                output_file = output_dir / file
                if output_file.exists():
                    LOGGER.info(f"{file} already exists; skipping!")
                    continue

                # Read the input audio file
                sr, audio_data = wavfile.read(str(input_file))

                # Split the audio into segments
                segments = split_audio(sr, audio_data)

                # Clear audio_data to free memory
                del audio_data

                # Process each segment
                for i, segment in enumerate(segments):
                    # Create temporary files for this segment
                    temp_input = temp_dir_path / f"input_{i}.wav"
                    temp_output = temp_dir_path / f"segment_{i}.wav"

                    # Write segment to temporary input file
                    wavfile.write(temp_input, sr, segment)

                    # Clear segment from memory
                    del segment

                    # Update config with temporary file paths
                    vc_single_config["input_audio_path"] = str(temp_input)

                    # Process the segment
                    success_string, wav_opt = vc.vc_single(**vc_single_config)
                    LOGGER.info(
                        f"Processed segment {i+1}/{len(segments)}: {success_string}"
                    )

                    # Write processed segment directly to temporary file
                    segment_sr, segment_data = wav_opt
                    wavfile.write(temp_output, segment_sr, segment_data)

                    # Clear wav_opt from memory
                    del wav_opt

                    # Remove input temporary file
                    temp_input.unlink()

                # Combine all processed segments
                try:
                    combine_audio_files(temp_dir_path, output_file)
                    LOGGER.info(
                        f"Successfully processed and combined {len(segments)} segments for {file}"
                    )
                finally:
                    # Clean up remaining temporary files
                    for temp_file in temp_dir_path.glob("segment_*.wav"):
                        temp_file.unlink()

    click.echo(f"Inference completed. Output saved to {output_dir}")


# env_path = Path(env_path)
# config_file = Path(config_file)
# input_audio_dir = Path(input_audio_dir)
# output_dir = Path(output_dir)

# vc_single_config = load_config(config_file)

# dotenv.load_dotenv(dotenv_path=env_path)
# model_name = os.getenv("model_name")
# config = Config()
# vc = API_VC(config)
# vc.get_vc(sid=model_name)

# os.makedirs(output_dir, exist_ok=True)

# for file in os.listdir(input_audio_dir):
#     if file.endswith(".wav"):
#         input_file = input_audio_dir / file
#         output_file = output_dir / file
#         vc_single_config["input_audio_path"] = str(input_file)

#         success_string, wav_opt = vc.vc_single(**vc_single_config)
#         LOGGER.info(success_string)

#         print(wav_opt)
#         sr, wav_opt = wav_opt
#         wavfile.write(output_file, sr, wav_opt)

# click.echo(f"Inference completed. Output saved to {output_dir}")


@cli.command()
@click.option(
    "--env-file",
    type=click.Path(exists=True),
    default=DEFAULT_ENV_PATH,
    help="Path to the .env file for training",
)
def train(env_file):
    """Train the voice conversion model."""
    os.environ["DOTENV_FILE"] = str(env_file)
    from rvc_cli.api.alternative_train import (
        API_train,
        create_3feature,
        create_f0_features,
        create_preprocessed_dataset,
        create_train_ctx,
    )

    click.echo("Starting training process...")
    create_preprocessed_dataset()
    create_f0_features()
    create_3feature()
    create_train_ctx()
    API_train()
    click.echo("Training completed.")


@cli.command()
@click.option(
    "--download-directory",
    type=click.Path(exists=True),
    help="Path of directory to download rvc models to (ie hubert.pt, etc)",
)
def download_rvc_models_to(download_directory: Path):
    """Download RVC models (hubert.pt, etc.)"""
    download_directory = Path(download_directory)
    click.echo(f"Starting download to {download_directory}...")
    download_rvc_models(download_directory=download_directory)
    click.echo("Download completed.")


@cli.command()
@click.option(
    "--input-file-path",
    help="Path to the input audio file",
)
@click.option(
    "--output-directory",
    help="Path to the output audio files",
)
@click.option("--chunk-size", help="Chunk size in seconds", default=60)
def create_dataset_from_large_mp3(input_file_path, output_directory, chunk_size):
    """Create as many 60 second clips from the provided mp3 file"""
    click.echo(
        f"Starting splitting of (potentially large) audio file to {output_directory}..."
    )
    process_audio_file(input_file_path, output_directory, chunk_size=chunk_size)
    click.echo("Splitting completed.")


if __name__ == "__main__":
    cli()
