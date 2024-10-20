import logging
import os
from typing import Optional

import torch

from rvc_cli.infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc_cli.infer.modules.vc.modules import VC
from rvc_cli.infer.modules.vc.pipeline import Pipeline

# import yaml


logger = logging.getLogger(__name__)


# def load_config():
#     config_path = os.getenv("CONFIG_PATH", "config.yml")
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#     return config


# idea: use a conditional, if no path for .env used, use the one in the home dir installed at runtime
#       else we take in a referenced one. this allows for if a user has set of models to do there stuff


class API_VC(VC):
    def get_vc(self, sid, *to_return_protect, index: Optional[str] = None):
        # Your new implementation here
        logger.info("Get sid: " + sid)
        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        # person = f'{os.getenv("weight_root")}/{sid}'
        # i switched to using exp_dir to be where we look

        # person = f'{os.getenv("weight_root")}/{sid}'
        person = f'{os.getenv("exp_dir")}/{sid}'
        logger.info(f"sid: {sid}")
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        if index:
            index = {
                "value": f"{index}",
                "__type__": "update",
            }
            logger.info("Select index: " + index["value"])
        else:
            index = None

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )


# def main():
#     """
#     config.yml is for passing into th vc.vc_single method with all keyword arguments
#     .env is for the pretrained_model_location, input_audio, and output_audio

#     cli usage:

#         CONFIG_PATH=config.yml python3 inference.py

#     api usage:
#     NOTE: may want to adjust the .yml and .env situation to give more flexibility

#         import os

#         from inference import API_VC,
#         from infer.configs.config import Config
#         from scipy.io import wavfile

#         model_name, opt_path = os.getenv("model_name"), os.getenv("output_audio")
#         config = Config()
#         vc = API_VC()
#         vc.get_vc(model_name)
#         success_string, wav_opt = vc.vc_single(**vc_single_config)
#         sr, wav_opt = wav_opt
#         wavfile.write(opt_path, sr, wav_opt)

#     """

#     model_name, opt_path = os.getenv("model_name"), os.getenv("output_audio")

#     config = Config()
#     vc = API_VC(config)
#     vc.get_vc(model_name)
#     success_string, wav_opt = vc.vc_single(**vc_single_config)
#     logging.info(success_string)
#     sr, wav_opt = wav_opt
#     wavfile.write(opt_path, sr, wav_opt)


# if __name__ == "__main__":
#     main()
