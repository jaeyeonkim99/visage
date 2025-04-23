import argparse
import glob
import json
import math
import os

import torch
import torchaudio
from audioldm_eval import EvaluationHelper
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")


def ambi2mono(ambisonics, elevation, azimuth):
    w = ambisonics[0]
    y = ambisonics[1]
    z = ambisonics[2]
    x = ambisonics[3]

    mono = (
        w
        + math.cos(elevation)
        * math.cos(azimuth)
        * x
        / +math.cos(elevation)
        * math.sin(azimuth)
        * y
        + math.sin(elevation) * z
    )

    return mono


def save_mono(wav, ambisonics, angle_dict, save_mono_dir, sr):
    base_name = os.path.splitext(wav)[0]
    elevation, azimuth = angle_dict[base_name]
    # Modify elevation, azimuth correctly
    modified_elevation = math.radians(89 - 2 * elevation)
    modified_azimuth = math.radians(azimuth * 2 - 179)
    mono_decoded = ambi2mono(ambisonics, modified_elevation, modified_azimuth)
    save_file_path = os.path.join(save_mono_dir, f"{base_name}.wav")

    torchaudio.save(save_file_path, mono_decoded.unsqueeze(0), sr, channels_first=True)


def metric_to_entry(metrics):
    fad = metrics["frechet_audio_distance"]
    kld = metrics["kullback_leibler_divergence_softmax"]

    return {"fad": f"{fad:.4f}", "kld": f"{kld:.4f}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_path",
        "-a",
        type=str,
        help="Path to the generated ambisonics",
    )
    parser.add_argument(
        "--result_path",
        "-r",
        type=str,
        default="metrics/result.json",
        help="Path the save the evaluation result",
    )
    parser.add_argument(
        "--test_samples",
        "-t",
        type=str,
        help="Path to the test samples compose of wav files of ambisonics, subdirectories of 'monodecode', 'w', 'y', 'z', and 'x'",
    )
    parser.add_argument(
        "--gt_angle_path",
        "-ga",
        type=str,
        default="csv/yt_ambigen/ambi.json",
        help="Path to ground truth camera direction for the videos",
    )

    args = parser.parse_args()

    print(f">Evaluating {args.audio_path}")
    wavs = glob.glob(os.path.join(args.audio_path, "*.wav"), recursive=False)

    with open(args.gt_angle_path, "r") as f:
        angle_dict = json.load(f)

    save_mono_dir = os.path.join(args.audio_path, "monodecode")
    os.makedirs(save_mono_dir, exist_ok=True)

    paths = ["w", "y", "z", "x"]
    save_dirs = [os.path.join(args.audio_path, path) for path in paths]

    evaluator = EvaluationHelper(16000, torch.device("cuda"))
    for dir in save_dirs:
        os.makedirs(dir, exist_ok=True)

    print("> Preparing audio files")
    for wav in tqdm(wavs, dynamic_ncols=True):
        audio, sr = torchaudio.load(wav, channels_first=True)
        if audio.shape[1] >= sr * 5:
            audio = audio[:, : sr * 5]
        else:
            pad_size = sr * 5 - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_size))

        audio = audio.float()
        file_name = os.path.basename(wav)
        save_mono(file_name, audio, angle_dict, save_mono_dir, sr)

        for idx, dir in enumerate(save_dirs):
            torchaudio.save(
                os.path.join(dir, file_name),
                audio[idx].unsqueeze(0),
                sr,
                channels_first=True,
            )

    if os.path.exists(args.result_path):
        with open(args.result_path) as f:
            results = json.load(f)
    else:
        results = {}

    to_eval = ["monodecode", "w", "y", "z", "x"]

    for idx, audio_type in enumerate(to_eval):
        print(f"> Evaluating {audio_type}...")
        metrics = evaluator.main(
            os.path.join(args.audio_path, audio_type),
            os.path.join(args.test_samples, audio_type),
            limit_num=None,
        )
        results[audio_type] = metric_to_entry(metrics)
    try:
        fads = []
        klds = []
        for entry in ["w", "y", "z", "x"]:
            fads.append(float(results[entry]["fad"]))
            klds.append(float(results[entry]["kld"]))
        results["fad_mean"] = sum(fads) / len(fads)
        results["kld_mean"] = sum(klds) / len(klds)
    except Exception as e:
        print(e)

    print(results)

    with open(args.result_path, "w") as f:
        json.dump(results, f, indent=2)
