import argparse
import glob
import os
import pdb

import clip
import ffmpeg
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_cossim_score(video_feat, height=7, width=7):
    # Get features for score computation
    temporal_feat = video_feat[:, 1:]
    temporal_feat = temporal_feat.view(-1, height, width, temporal_feat.shape[-1])

    # Compute local score
    local_feat = F.normalize(temporal_feat, p=2, dim=-1)

    # Compute spatial score
    M = torch.zeros_like(temporal_feat)
    C = torch.zeros_like(temporal_feat)
    M[:, 1:] += temporal_feat[:, :-1]
    M[:, :-1] += temporal_feat[:, 1:]
    M[:, :, 1:] += temporal_feat[:, :, :-1]
    M[:, :, :-1] += temporal_feat[:, :, 1:]
    C[:, 1:] += 1
    C[:, :-1] += 1
    C[:, :, 1:] += 1
    C[:, :, :-1] += 1
    spatial_feat = F.normalize(M / C, p=2, dim=-1)
    spaital_score = torch.norm(local_feat - spatial_feat, p=2, dim=-1)

    # Compute temporal score
    # temporal feat: [T(B), H, W, C]
    temporal_avg = temporal_feat.detach().clone()
    temporal_avg[1:] = temporal_feat.detach()[:-1]
    temporal_avg[0] = 0.0
    temporal_avg[:-1] += temporal_feat.detach()[1:]
    temporal_avg[1:-1] /= 2.0
    temporal_avg = F.normalize(temporal_avg, p=2, dim=-1)
    temporal_score = torch.norm(local_feat - temporal_avg, p=2, dim=-1)

    return spaital_score, temporal_score


def compute_energy_map(scores, temperature=0.1):
    b, h, w = scores[0].shape
    average_energy_map = torch.zeros(b, h * w).to(scores[0].device)
    for score in scores:
        score = score / temperature
        energy_map = score.view(b, -1).softmax(-1)
        average_energy_map += energy_map

    average_energy_map /= len(scores)

    return average_energy_map


def top_p_filtering(energy_map, scores, top_p=0.5, filter_value=-float("Inf")):
    sorted_energy, sorted_indices = torch.sort(energy_map, descending=False)
    cumulative_energy = torch.cumsum(sorted_energy, dim=-1)
    sorted_indices_to_remove = cumulative_energy <= (1 - top_p)

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    b, h, w = scores[0].shape
    processed_scores = []
    for score in scores:
        score = score.view(b, -1)
        score_processed = score.masked_fill(indices_to_remove, filter_value)
        score_processed = score_processed.view(b, h, w)
        processed_scores.append(score_processed)

    return processed_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_patch_path",
        "-cp",
        type=str,
        help="Path to preprocessed clip patch level features",
    )
    parser.add_argument(
        "--save_path", "-s", type=str, help="Path to save the energy map features"
    )
    args = parser.parse_args()

    video_files = glob.glob(
        os.path.join(args.clip_patch_path, "**/*.npy"), recursive=True
    )
    print_shape = True

    with torch.no_grad():
        for video in tqdm(video_files, dynamic_ncols=True):
            vid = os.path.splitext(os.path.basename(video))[0]
            save_path = os.path.join(args.save_path, vid + ".npy")
            if os.path.exists(save_path):
                continue

            clip_path = os.path.join(args.clip_patch_path, vid + ".npy")
            video_feat = torch.from_numpy(np.load(clip_path)).cuda()
            if print_shape:
                print("CLIP patchwise feature:", video_feat.shape)
            if video_feat.shape[0] == 0:
                print("No temporal dimension for ", vid)
                continue

            scores = compute_cossim_score(video_feat)
            energy_map = compute_energy_map(scores, temperature=0.1)
            filtered_scores = top_p_filtering(energy_map, scores, 0.7)
            filtered_energy_map = compute_energy_map(filtered_scores)

            if print_shape:
                print("Energy map: ", filtered_energy_map.shape)
                print_shape = False

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, filtered_energy_map.detach().cpu().numpy())
