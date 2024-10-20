import logging
import os
from pathlib import Path

import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def dl_model(link, model_name, dir_name):
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dir_name / model_name), exist_ok=True)
        with open(dir_name / model_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_rvc_models(download_directory: Path):
    LOGGER.info("Downloading hubert_base.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "hubert_base.pt", download_directory / "assets/hubert")
    LOGGER.info("Downloading rmvpe.pt...")
    dl_model(RVC_DOWNLOAD_LINK, "rmvpe.pt", download_directory / "assets/rmvpe")
    LOGGER.info("Downloading vocals.onnx...")
    dl_model(
        RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/",
        "vocals.onnx",
        download_directory / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy",
    )

    rvc_models_dir = download_directory / "assets/pretrained"

    LOGGER.info("Downloading pretrained models:")

    model_names = [
        "D32k.pth",
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]
    for model in model_names:
        LOGGER.info(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "pretrained/", model, rvc_models_dir)

    rvc_models_dir = download_directory / "assets/pretrained_v2"

    LOGGER.info("Downloading pretrained models v2:")

    for model in model_names:
        LOGGER.info(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "pretrained_v2/", model, rvc_models_dir)

    LOGGER.info("Downloading uvr5_weights:")

    rvc_models_dir = download_directory / "assets/uvr5_weights"

    model_names = [
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]
    for model in model_names:
        LOGGER.info(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK + "uvr5_weights/", model, rvc_models_dir)

    LOGGER.info("All models downloaded!")
