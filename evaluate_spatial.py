import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from utils.view_video import Position, spherical_grid, spherical_harmonics_matrix

# Only import AUC_Judd and CorrCoeff
from utils.wild_360 import AUC_Judd, CorrCoeff  # Removed AUC_Borji, similarity


def dict_to_score(dict_with_scores):
    dict_total = list(dict_with_scores.values())
    score = sum(dict_total) / len(dict_total)

    return score


def er_sphere_sample(data):
    H, W = data.shape[0], data.shape[1]
    out_data = np.zeros_like(data)
    out = list()
    for ele in range(H):
        R = np.sin(np.pi * (ele + 0.5) / H)
        S = np.round(W * R)
        sample = np.interp(
            x=np.arange(S) * (W - 0.5 * (W / S)) / S,
            xp=np.arange(W),
            fp=data[ele],
            period=360,
        )
        out_data[
            ele,
            W // 2
            - sample.shape[0] // 2 : W // 2
            - sample.shape[0] // 2
            + sample.shape[0],
        ] = sample
        out.extend(sample.tolist())

    return np.array(out)


class AmbiDecomposition(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-computed spherical grid and harmonics matrix
        self.phi_mesh, self.nu_mesh = spherical_grid(2.0)
        self.mesh_p = [
            Position(phi, nu, 1.0)
            for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))
        ]
        matrix = torch.from_numpy(spherical_harmonics_matrix(self.mesh_p, 1).T).float()
        self.register_buffer("matrix", matrix)
        self.out_shape = (-1, self.phi_mesh.shape[0], self.phi_mesh.shape[1])

    def get_argmax_idx(self, x):
        return [int(x) for x in np.unravel_index(np.argmax(x, axis=None), x.shape)]

    def get_angle_from_map(self, maps):
        return [self.get_argmax_idx(x) for x in maps]

    @torch.no_grad()
    def forward(self, x):
        out = torch.einsum("btc,ch->bth", x, self.matrix)
        out = (out**2).mean(1).sqrt().reshape(self.out_shape)  # BAE
        return [self.get_argmax_idx(np.flipud(x)) for x in out.cpu().numpy()]

    @torch.no_grad()
    def get_energy_map(self, x):
        out = torch.einsum("btc,ch->bth", x, self.matrix)
        out = (out**2).mean(1).sqrt().reshape(self.out_shape)  # BAE
        return [np.flipud(x) for x in out.cpu().numpy()]


def process_overall(data):
    """
    Processes the overall calculation for full video/audio metrics.
    This function will be executed in parallel.
    """
    file_id, energy_map, gt_energy_map = data

    corr = CorrCoeff(energy_map.copy(), gt_energy_map.copy())
    auc_j = AUC_Judd(energy_map.copy(), gt_energy_map.copy())

    return (file_id, corr, auc_j)


def process_fps(data):
    """
    Processes for a given FPS (1fps or 5fps).
    This function will be executed in parallel to calculate the 1fps and 5fps metrics.
    """
    file_id, energy_map_fps, gt_energy_map_fps, fps = data

    # Store metrics
    corrs_fps = []
    auc_j_fps = []

    for i in range(len(energy_map_fps)):
        try:
            corr = CorrCoeff(energy_map_fps[i].copy(), gt_energy_map_fps[i].copy())
            auc_j = AUC_Judd(energy_map_fps[i].copy(), gt_energy_map_fps[i].copy())
        except Exception as e:
            print(e)
            print(f"Error processing during {file_id}, Skipping...")
            continue

        if not np.isnan(corr) and not np.isnan(auc_j):
            corrs_fps.append(corr)
            auc_j_fps.append(auc_j)
        else:
            print(f"Skipping {file_id} due to NaN in {fps}fps setting")

    # Calculate the average across frames
    avg_corr = sum(corrs_fps) / len(corrs_fps)
    avg_auc_j = sum(auc_j_fps) / len(auc_j_fps)

    return (file_id, fps, avg_corr, avg_auc_j)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_path", "-a", type=str, help="Path of the generated audio files"
    )
    parser.add_argument(
        "--result_path",
        "-r",
        type=str,
        default="metrics/result.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--gt_audio_map_path",
        "-ga",
        type=str,
        help="Path to ground truth audio energy maps",
    )

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)

    print(f"> Running spatial evaluation for {args.audio_path}...")

    wavs = glob.glob(os.path.join(args.audio_path, "**.wav"), recursive=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decom = AmbiDecomposition().to(device)

    # Initialize result dictionaries
    corr_dict = {}
    auc_j_dict = {}
    corr_1_dict = {}
    auc_j_1_dict = {}
    corr_5_dict = {}
    auc_j_5_dict = {}

    # Prepare lists for parallel processing
    fps_data_list = []
    overall_data_list = []

    print("> Preparing Energy Maps")
    for wav in tqdm(wavs, dynamic_ncols=True):
        file_name = os.path.basename(wav)
        file_id = os.path.splitext(file_name)[0]
        gt = os.path.join(args.gt_audio_map_path, file_id + ".npz")
        gt_data_npz = np.load(gt)

        audio, sr = torchaudio.load(wav, channels_first=True)
        if audio.shape[1] >= sr * 5:
            audio = audio[:, : sr * 5]
        else:
            pad_size = sr * 5 - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        audio = audio.transpose(0, 1).unsqueeze(0).float().to(device)

        # Get the energy map for the file
        maps = decom.get_energy_map(audio)
        energy_map = maps[0]
        energy_map = er_sphere_sample(energy_map)
        gt_energy_map = gt_data_npz["map"]

        # Prepare data for multiprocessing (overall metrics)
        overall_data_list.append((file_id, energy_map, gt_energy_map))

        # 1fps case
        audio_1fps = audio.view(-1, sr, audio.shape[-1])
        energy_map_1fps = decom.get_energy_map(audio_1fps)
        for i in range(len(energy_map_1fps)):
            energy_map_1fps[i] = er_sphere_sample(energy_map_1fps[i])
        gt_energy_map_1fps = gt_data_npz["map_1fps"]

        # 5fps case
        audio_5fps = audio.view(-1, sr // 5, audio.shape[-1])
        energy_map_5fps = decom.get_energy_map(audio_5fps)
        for i in range(len(energy_map_5fps)):
            energy_map_5fps[i] = er_sphere_sample(energy_map_5fps[i])
        gt_energy_map_5fps = gt_data_npz["map_5fps"]

        # Collect data for fps-specific calculations
        fps_data_list.append((file_id, energy_map_1fps, gt_energy_map_1fps, 1))
        fps_data_list.append((file_id, energy_map_5fps, gt_energy_map_5fps, 5))

    # Parallelize overall calculation
    print("> Calculating Overall Metrics")
    with Pool(cpu_count()) as pool:
        overall_results = list(
            tqdm(
                pool.imap(process_overall, overall_data_list),
                total=len(overall_data_list),
            )
        )

    # Collect overall results
    for result in overall_results:
        file_id, corr, auc_j = result
        corr_dict[file_id] = corr
        auc_j_dict[file_id] = auc_j

    corr_result = dict_to_score(corr_dict)
    auc_j_result = dict_to_score(auc_j_dict)

    print(f"Corr overall: {corr_result}, AUC_Judd overall: {auc_j_result}")

    # Parallelize FPS-specific calculations
    print("> Calculating fps Metrics")
    with Pool(cpu_count()) as pool:
        fps_results = list(
            tqdm(pool.imap(process_fps, fps_data_list), total=len(fps_data_list))
        )

    # Collect FPS results
    for result in fps_results:
        file_id, fps, avg_corr, avg_auc_j = result
        if fps == 1:
            corr_1_dict[file_id] = avg_corr
            auc_j_1_dict[file_id] = avg_auc_j
        elif fps == 5:
            corr_5_dict[file_id] = avg_corr
            auc_j_5_dict[file_id] = avg_auc_j

    # Save individual-file dictionaries
    file_dicts = [
        ("corr_dict.json", corr_dict),
        ("auc_j_dict.json", auc_j_dict),
        ("corr_1_dict.json", corr_1_dict),
        ("auc_j_1_dict.json", auc_j_1_dict),
        ("corr_5_dict.json", corr_5_dict),
        ("auc_j_5_dict.json", auc_j_5_dict),
    ]

    try:
        for file_name, data_dict in file_dicts:
            with open(os.path.join(args.audio_path, file_name), "w") as f:
                json.dump(data_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(e)

    # Compute final overall metrics
    corr_result = dict_to_score(corr_dict)
    auc_j_result = dict_to_score(auc_j_dict)
    corr_1_result = dict_to_score(corr_1_dict)
    auc_j_1_result = dict_to_score(auc_j_1_dict)
    corr_5_result = dict_to_score(corr_5_dict)
    auc_j_5_result = dict_to_score(auc_j_5_dict)

    # Either load or create new results JSON
    if os.path.exists(args.result_path):
        with open(args.result_path) as f:
            results = json.load(f)
    else:
        results = {}

    # Update and save final result metrics
    try:
        with open(args.result_path, "w") as f:
            results_new = {
                "corr_all": f"{corr_result:.4f}",
                "auc_j_all": f"{auc_j_result:.4f}",
                "corr_1fps": f"{corr_1_result:.4f}",
                "auc_j_1fps": f"{auc_j_1_result:.4f}",
                "corr_5fps": f"{corr_5_result:.4f}",
                "auc_j_5fps": f"{auc_j_5_result:.4f}",
            }
            results.update(results_new)
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(e)
        print(results)

    print(results)
