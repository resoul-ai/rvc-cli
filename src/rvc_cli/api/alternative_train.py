import json
import logging
import os
import pathlib
import shutil
import sys
import warnings
from random import randint, shuffle

import torch
from dotenv import load_dotenv

load_dotenv(os.getenv("DOTENV_FILE"))

from rvc_cli.configs.config import Config
from rvc_cli.infer.lib.audio import load_audio
from rvc_cli.infer.lib.train import utils
from rvc_cli.infer.modules.train.extract.extract_f0_rmvpe import FeatureInput
from rvc_cli.infer.modules.train.preprocess import preprocess_trainset
from rvc_cli.infer.modules.train.train import run
from rvc_cli.infer.modules.vc.modules import VC

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(randint(20000, 55555))

# we naively assume tmp is needed
now_dir = os.getenv("root_dir")
sys.path.append(os.path.join(now_dir))
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

if torch.cuda.is_available():
    GPU = torch.cuda.get_device_name(0)
    logger.warning(f"hard set to only use single GPU: {GPU}")
else:
    raise Exception("No GPU found")

config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def create_preprocessed_dataset(
    inp_root: str = os.getenv("input_root"),
    sr: int = 40000,
    n_p: int = 14,
    exp_dir: str = os.getenv("exp_dir"),
    noparallel: bool = True,
    per: int = 3.7,
):
    # NOTE: not sure what noparallel was for
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, noparallel)


class API_FeatureInput(FeatureInput):
    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        from rvc_cli.infer.lib.rmvpe import RMVPE

        self.model_rmvpe = RMVPE(
            os.path.join(os.getenv("rmvpe_root"), "rmvpe.pt"),
            is_half=os.getenv("rmvpe_is_half"),
            device="cuda",
        )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        return f0


def create_f0_features(
    exp_dir: str = os.getenv("exp_dir"),
):
    # set up feature input
    featureInput = API_FeatureInput()
    paths = []

    # set up paths
    os.makedirs(exp_dir, exist_ok=True)
    inp_root = "%s/1_16k_wavs" % (exp_dir)

    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    # fill paths
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])

    # run feature input
    # in original webui they use cmd and allocate to diff gpus, we assume 1
    featureInput.go(paths, "rmvpe")


def create_3feature():
    # import subprocess

    # # the extract_feature_print is a convoluted
    # # script that uses hubert. I should port to here but
    # # for now we'll just run it
    # # TODO: port extract_feature_print to here
    # # here's how, just put anything in an else block from len(sys.argv) > 1 here + core fns
    # subprocess.run(
    #     [
    #         "python",
    #         "/home/arelius/workspace/utter/Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/extract_feature_print.py",
    #     ]
    # )

    # testing to see if this works
    import os
    import sys
    import traceback

    import dotenv

    dotenv.load_dotenv()

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # webuui?
    print(sys.argv)
    # if len(sys.argv) > 1:
    #     device = sys.argv[1]
    #     n_part = int(sys.argv[2])
    #     i_part = int(sys.argv[3])
    #     if len(sys.argv) == 7:
    #         exp_dir = sys.argv[4]
    #         version = sys.argv[5]
    #         is_half = sys.argv[6].lower() == "true"
    #     else:
    #         i_gpu = sys.argv[4]
    #         exp_dir = sys.argv[5]
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    #         version = sys.argv[6]
    #         is_half = sys.argv[7].lower() == "true"

    # else:
    #     exp_dir = os.getenv("exp_dir")

    # lets just guess and see
    exp_dir = os.getenv("exp_dir")
    device = os.getenv("device")
    # it is documented nowhere what on earth n_part and i_part are. i could extract via uncommenting arparse stuff but will not rn
    n_part = 1
    i_part = 0
    version = "v2"
    is_half = "true"

    import fairseq
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    # if len(sys.argv) > 1:
    #     if "privateuseone" not in device:
    #         device = "cpu"
    #         if torch.cuda.is_available():
    #             device = "cuda"
    #         elif torch.backends.mps.is_available():
    #             device = "mps"
    #     else:
    #         import torch_directml
    #         device = torch_directml.device(torch_directml.default_device())
    #         def forward_dml(ctx, x, scale):
    #             ctx.scale = scale
    #             res = x.clone().detach()
    #             return res
    #         fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
    # else:
    #     device = "cpu"
    #     if torch.cuda.is_available():
    #         device = "cuda"
    #     elif torch.backends.mps.is_available():
    #         device = "mps"
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    f = open("%s/extract_f0_feature.log" % exp_dir, "a+")

    def printt(strr):
        print(strr)
        f.write("%s\n" % strr)
        f.flush()

    # if len(sys.argv) > 1:
    #     printt(" ".join(sys.argv))
    #     model_path = "assets/hubert/hubert_base.pt"
    # else:
    #     model_path = os.getenv("hubert_base_path")
    model_path = os.getenv("hubert_base_path")

    printt("exp_dir: " + exp_dir)
    wavPath = "%s/1_16k_wavs" % exp_dir
    # if len(sys.argv) > 1:
    #     outPath = (
    #         "%s/3_feature256" % exp_dir
    #         if version == "v1"
    #         else "%s/3_feature768" % exp_dir
    #     )
    #     os.makedirs(outPath, exist_ok=True)
    # else:
    # version = "v2"
    # outPath = (
    #     "%s/3_feature256" % exp_dir
    #     if version == "v1"
    #     else "%s/3_feature768" % exp_dir
    # )
    # os.makedirs(outPath, exist_ok=True)
    version = "v2"
    outPath = (
        "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
    )
    os.makedirs(outPath, exist_ok=True)

    # wave must be 16k, hop_size=320
    def readwave(wav_path, normalize=False):
        wav, sr = sf.read(wav_path)
        assert sr == 16000
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        feats = feats.view(1, -1)
        return feats

    # HuBERT model
    printt("load model(s) from {}".format(model_path))
    # if hubert model is exist
    if os.access(model_path, os.F_OK) == False:
        printt(
            "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            % model_path
        )
        exit(0)
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    model = models[0]
    model = model.to(device)
    printt("move model to %s" % device)
    if len(sys.argv) > 1:
        if is_half:
            if device not in ["mps", "cpu"]:
                model = model.half()
    else:
        is_half = True
        if is_half:
            if device not in ["mps", "cpu"]:
                model = model.half()

    model.eval()

    if len(sys.argv) > 1:
        todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
    else:
        todo = sorted(list(os.listdir(wavPath)))
    n = max(1, len(todo) // 10)  # 最多打印十条
    if len(todo) == 0:
        printt("no-feature-todo")
    else:
        printt("all-feature-%s" % len(todo))
        for idx, file in enumerate(todo):
            try:
                if file.endswith(".wav"):
                    wav_path = "%s/%s" % (wavPath, file)
                    out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                    if os.path.exists(out_path):
                        continue

                    feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": (
                            feats.half().to(device)
                            if is_half and device not in ["mps", "cpu"]
                            else feats.to(device)
                        ),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 9 if version == "v1" else 12,  # layer 9
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = (
                            model.final_proj(logits[0])
                            if version == "v1"
                            else logits[0]
                        )

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        printt("%s-contains nan" % file)
                    if idx % n == 0:
                        printt(
                            "now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape)
                        )
            except:
                printt(traceback.format_exc())
        printt("all-feature-done")


def create_train_ctx(
    if_f0_3: bool = True,  # we had if_f0_3 true
    spk_id5: int = 0,  # we had spk_id5 0
    version19: str = "v2",  # we had version19 v2
    sr2: str = "40k",  # we had sr2 40k
    exp_dir: str = os.getenv("exp_dir"),
    gt_wavs_dir: str = os.path.join(os.getenv("exp_dir"), "0_gt_wavs"),
    feature_dir: str = os.path.join(os.getenv("exp_dir"), "3_feature768"),
    f0_dir: str = os.path.join(os.getenv("exp_dir"), "2a_f0"),
    f0nsf_dir: str = os.path.join(os.getenv("exp_dir"), "2b-f0nsf"),
    # gt_wavs_dir: str = os.getenv("exp_dir").join("/0_gt_wavs"),
    # feature_dir: str = os.getenv("exp_dir").join("/3_feature768"),  # v2==768
    # f0_dir: str = os.getenv("exp_dir").join("/2b-f0"),
    # f0nsf_dir: str = os.getenv("exp_dir").join("/2b-f0nsf"),
):
    names = (
        set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
        & set([name.split(".")[0] for name in os.listdir(feature_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
    )

    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")

    # we used v2 initially
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2

    config_base_path = os.getenv("config_base_path")
    config_path = os.path.join(config_base_path, config_path)
    with open(config_path, "r") as f:
        d = json.load(f)

    config_save_path = os.path.join(exp_dir, "config.json")

    if (
        not pathlib.Path(config_save_path).exists()
        or os.stat(config_save_path).st_size == 0
    ):
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                d,
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")


def API_train(
    # these are all inputs to get_hparams
    exp_dir: str = os.getenv("exp_dir"),
    sr: str = "40k",
    if_f0_3: int = 1,
    batch_size: int = 12,
    gpu: int = 1,
    total_epoch: int = 20,
    save_every_epoch: int = 5,
    pretrainG: str = os.getenv("pretrained_G"),
    pretrainD: str = os.getenv("pretrained_D"),
    if_save_latest: int = 0,
    if_cache_data_in_gpu: int = 0,
    if_save_every_weights: int = 0,
    version: str = "v2",
):
    # uses a config file to load into hparams
    # /home/arelius/workspace/utter/Retrieval-based-Voice-Conversion-WebUI/logs/webui-test/config.json
    # NOTE: how did they originally generate this?
    config_save_path = os.path.join(exp_dir, "config.json")
    with open(config_save_path, "r") as f:
        config = json.load(f)

    # subsequently adds the above init values to the hparams
    hparams = utils.HParams(**config)

    # returns hparams
    hparams.model_dir = exp_dir
    hparams.save_every_epoch = save_every_epoch
    hparams.name = exp_dir.split("/")[-1]
    hparams.total_epoch = total_epoch
    hparams.pretrainG = pretrainG
    hparams.pretrainD = pretrainD
    hparams.version = version
    hparams.gpus = gpu
    hparams.train.batch_size = batch_size
    hparams.sample_rate = sr
    hparams.if_f0 = if_f0_3
    hparams.if_latest = if_save_latest
    hparams.save_every_weights = if_save_every_weights
    hparams.if_cache_data_in_gpu = if_cache_data_in_gpu
    hparams.data.training_files = "%s/filelist.txt" % exp_dir

    run(rank=0, n_gpus=1, hps=hparams, logger=logger)


def main():
    """
    .env is for the pretrained_model_location, input_audio, and output_audio

    cli usage:

        python3 alternative_train.py

    api usage:

        from alternative_train import create_preprocessed_dataset, create_f0_features, create_3feature, create_train_ctx, API_train
        # choose input directory, exp directory, and anything else (otherwise it will default to the .env file)

        create_preprocessed_dataset()
        create_f0_features()
        create_3feature()
        create_train_ctx()
        API_train()


    """
    create_preprocessed_dataset()
    create_f0_features()
    create_3feature()
    create_train_ctx()
    # NOTE: a question remains, are small models always extracted at
    # each save or not? clearly they are after full training run
    API_train()


# if __name__ == "__main__":
# main()

# set (1)
# experiment name
# sample rate
# whether model has pitch guidance (yes)
# version (v2)
# number of cpu processes (14)

# perform data preprocessing (2a)
# set training path folder
# specify speaker id (0)
# run process fn

# perform pitch extraction (2b)
# set gpu indices (0)
# set gpu info (autofill with an os check probably)
# set pitch extraction algorithm (rmvpe_gpu)
# run feature extraction fn

# perform training (3) (we don't train a feature index)
# set save_every_epoch (5)
# set total_epoch (20)
# set batch sizer per GPU (12)
# set save only latest '.ckpt' file (No)
# set Cache all training sets to GPU memory (No)
# set save a small model for inference in weights/ (Yes)
# set load pre-trained model G path (assets/pretrained_v2/f0G40k.pth)
# set load pre-trained model D path (assets/pretrained_v2/f0D40k.pth)
# run train model
# run train model
