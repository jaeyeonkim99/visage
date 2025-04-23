import argparse
import json
import math
import os
import pdb

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm


def ambi2mono(ambisonics, elevation, azimuth):
    w = ambisonics[0]
    y = ambisonics[1]
    z = ambisonics[2]
    x = ambisonics[3]

    mono = (
        w
        + math.cos(elevation) * math.cos(azimuth) * x
        + math.cos(elevation) * math.sin(azimuth) * y
        + math.sin(elevation) * z
    )

    return mono


def save_mono(wav_name, ambisonics, angle_dict, save_mono_dir, sr):
    elevation, azimuth = angle_dict[wav_name]
    # Modify elevation, azimuth correctly
    modified_elevation = math.radians(89 - 2 * elevation)
    modified_azimuth = math.radians(azimuth * 2 - 179)
    mono_decoded = ambi2mono(ambisonics, modified_elevation, modified_azimuth)
    save_file_path = os.path.join(save_mono_dir, f"{wav_name}.wav")

    torchaudio.save(save_file_path, mono_decoded.unsqueeze(0), sr, channels_first=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        help="Path to the original audio files",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        default="test_audio",
        help="Path to save the processed test audios",
    )
    parser.add_argument(
        "--test_csv",
        "-tc",
        type=str,
        default="../csv/yt_ambigen/test.csv",
        help="CSV file with list of test files for evaluation",
    )
    parser.add_argument(
        "--audio_extension",
        "-ae",
        type=str,
        default=".opus",
        help="Audio file extension for original audio files",
    )
    parser.add_argument(
        "--angle_dict",
        "-ad",
        type=str,
        default="../csv/yt_ambigen/ambi.json",
        help="Path to the gt camera direction",
    )
    args = parser.parse_args()

    test_files = pd.read_csv(args.test_csv)["file_path"].tolist()
    test_wavs = [
        os.path.join(args.data_path, path.replace(".npy", args.audio_extension))
        for path in test_files
    ]
    mono_dir = os.path.join(args.save_path, "monodecode")
    os.makedirs(mono_dir, exist_ok=True)
    save_dirs = []

    for channel in ["w", "y", "z", "x"]:
        save_dir = os.path.join(args.save_path, channel)
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)

    with open(args.angle_dict) as f:
        angle_dict = json.load(f)

    for wav in tqdm(test_wavs, dynamic_ncols=True, desc="Processing wavs"):
        audio, sr = sf.read(wav)
        wav_name = os.path.basename(wav)
        wav_name = os.path.splitext(wav_name)[0]

        # Resample
        audio = torch.from_numpy(audio).transpose(0, 1).float()
        resampled = resample(audio, sr, 16000)
        if resampled.shape[1] >= 16000 * 5:
            resampled = resampled[:, : 16000 * 5]
        else:
            pad_size = 16000 * 5 - resampled.shape[1]
            resampled = torch.nn.functional.pad(resampled, (0, pad_size))

        # Save the processed audio
        save_file_path = os.path.join(args.save_path, f"{wav_name}.wav")
        torchaudio.save(save_file_path, resampled, 16000, channels_first=True)

        # Save mono decode
        save_mono(wav_name, resampled, angle_dict, mono_dir, 16000)

        # Save each file
        for idx, dir in enumerate(save_dirs):
            torchaudio.save(
                os.path.join(dir, f"{wav_name}.wav"),
                resampled[idx].unsqueeze(0),
                16000,
                channels_first=True,
            )
