import argparse
import glob
import os
import sys

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from evaluate_spatial import AmbiDecomposition, er_sphere_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ground truth energy map from ambisonics audio"
    )
    parser.add_argument(
        "--gt_audio_path",
        type=str,
        default="test_audio",
        help="Path to the directory containing ground truth audio files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="gt_energy_map",
        help="Path to save the energy map files",
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    wavs = glob.glob(os.path.join(args.gt_audio_path, "*.wav"), recursive=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decom = AmbiDecomposition().to(device)

    for wav in tqdm(wavs, dynamic_ncols=True):
        file_name = os.path.basename(wav)
        file_id = os.path.splitext(file_name)[0]

        gt_audio, sr = torchaudio.load(wav, channels_first=True)
        if gt_audio.shape[1] >= sr * 5:
            gt_audio = gt_audio[:, : sr * 5]
        else:
            pad_size = sr * 5 - gt_audio.shape[1]
            gt_audio = torch.nn.functional.pad(gt_audio, (0, pad_size))

        # [B, T, C] to get energy map
        gt_audio = gt_audio.transpose(0, 1).unsqueeze(0).float().to(device)
        gt_energy_map = decom.get_energy_map(gt_audio)[0]
        gt_energy_map = er_sphere_sample(gt_energy_map)

        gt_audio_1fps = gt_audio.view(-1, sr, gt_audio.shape[-1])
        gt_energy_map_1fps = decom.get_energy_map(gt_audio_1fps)
        for i in range(len(gt_energy_map_1fps)):
            gt_energy_map_1fps[i] = er_sphere_sample(gt_energy_map_1fps[i])
        gt_energy_map_1fps = np.array(gt_energy_map_1fps)

        gt_audio_5fps = gt_audio.view(-1, sr // 5, gt_audio.shape[-1])
        gt_energy_map_5fps = decom.get_energy_map(gt_audio_5fps)
        for i in range(len(gt_energy_map_5fps)):
            gt_energy_map_5fps[i] = er_sphere_sample(gt_energy_map_5fps[i])
        gt_energy_map_5fps = np.array(gt_energy_map_5fps)

        save_file = os.path.join(args.save_path, file_id + ".npz")
        np.savez(
            save_file,
            map=gt_energy_map,
            map_1fps=gt_energy_map_1fps,
            map_5fps=gt_energy_map_5fps,
        )
