import argparse
import glob
import json
import math
import os
import pdb

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def mono2ambi(w_channel, elevation, azimuth):
    s = math.sqrt(2) * w_channel
    x_channel = s * math.cos(elevation) * math.cos(azimuth)
    y_channel = s * math.cos(elevation) * math.sin(azimuth)
    z_channel = s * math.sin(elevation)

    ambisonics = torch.stack([w_channel, y_channel, z_channel, x_channel], axis=0)

    return ambisonics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated_audio_path",
        "-g",
        type=str,
        help="Path to the generated audio files",
    )
    parser.add_argument(
        "--save_ambisonics_path",
        "-s",
        type=str,
        help="Path to save the encoded ambisonics files",
    )
    parser.add_argument(
        "--gt_angle_path",
        "-ga",
        type=str,
        default="csv/yt_ambigen/ambi.json",
        help="Path to ground truth camera direction for the videos",
    )

    args = parser.parse_args()

    generated_audios = glob.glob(
        os.path.join(args.generated_audio_path, "**/*.wav"), recursive=True
    )
    os.makedirs(args.save_ambisonics_path, exist_ok=True)
    with open(args.gt_angle_path) as f:
        gt_angle_dict = json.load(f)

    for audio_files in tqdm(generated_audios):
        file_name = os.path.basename(audio_files).replace(".wav", "")
        if file_name in gt_angle_dict:
            elevation, azimuth = gt_angle_dict[file_name]

            # Modify elevation, azimuth correctly
            modified_elevation = math.radians(89 - 2 * elevation)
            modified_azimuth = math.radians(azimuth * 2 - 179)

            audio, sr = torchaudio.load(audio_files)
            ambisonics = mono2ambi(audio[0], modified_elevation, modified_azimuth)
            save_file_path = os.path.join(args.save_ambisonics_path, f"{file_name}.wav")
            torchaudio.save(save_file_path, ambisonics, sr, channels_first=True)
