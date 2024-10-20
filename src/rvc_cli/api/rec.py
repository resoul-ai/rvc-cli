import logging
import os

from scipy.io import wavfile

from rvc_cli.api.inference import API_VC  # , vc_single_config
from rvc_cli.configs.config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train():
    """Adjust .env before calling

    The training updates are currently stored under
    /home/arelius/workspace/utter/Retrieval-based-Voice-Conversion-WebUI/logs

    hparam info to be passed to API_train()
    """
    os.environ["DOTENV_FILE"] = (
        "/home/arelius/workspace/audible.vc/data/v2v/defaults/.env"
    )
    from rvc_cli.api.alternative_train import (  # dotenv.load_dotenv("/home/arelius/workspace/audible.vc/data/v2v/defaults/.env"); should probably pass an argument of .env location for everything to consume
        API_train,
        create_3feature,
        create_f0_features,
        create_preprocessed_dataset,
        create_train_ctx,
    )

    create_preprocessed_dataset()
    create_f0_features()
    create_3feature()
    create_train_ctx()
    # adding in .env file location
    API_train()


def infer():
    """Adjust .env before calling

    Requires a flag to config.yml path:
    CONFIG_PATH=config.yml python3 rec.py

    yml contains inference defaults and audio sample to mimic (v2v)
    """
    model_name, _opt_path = os.getenv("model_name"), os.getenv("output_audio")
    input_audio_path, output_path = os.getenv("INPUT"), os.getenv("OUTPUT")

    config = Config()
    vc = API_VC(config)
    vc.get_vc(model_name)
    # vc_single_config["input_audio_path"] = input_audio_path
    # success_string, wav_opt = vc.vc_single(**vc_single_config)
    vc.vc_single_config["input_audio_path"] = input_audio_path
    success_string, wav_opt = vc.vc_single(**vc.vc_single_config)
    logging.info(success_string)
    print(wav_opt)
    sr, wav_opt = wav_opt
    wavfile.write(output_path, sr, wav_opt)


def main():
    logging.getLogger(__name__)
    # infer()
    train()


if __name__ == "__main__":
    main()
