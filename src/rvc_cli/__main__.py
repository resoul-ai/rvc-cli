import logging
import os
from pathlib import Path

import click
import dotenv
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

    for file in os.listdir(input_audio_dir):
        if file.endswith(".wav"):
            input_file = input_audio_dir / file
            output_file = output_dir / file
            vc_single_config["input_audio_path"] = str(input_file)

            success_string, wav_opt = vc.vc_single(**vc_single_config)
            LOGGER.info(success_string)

            print(wav_opt)
            sr, wav_opt = wav_opt
            wavfile.write(output_file, sr, wav_opt)

    click.echo(f"Inference completed. Output saved to {output_dir}")


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
def create_dataset_from_large_mp3(input_file_path, output_directory):
    """Create as many 60 second clips from the provided mp3 file"""
    click.echo(
        f"Starting splitting of (potentially large) audio file to {output_directory}..."
    )
    process_audio_file(input_file_path, output_directory, chunk_size=60)
    click.echo("Splitting completed.")


if __name__ == "__main__":
    cli()
