import json
import logging
import pdb
import random
from dataclasses import dataclass
from pathlib import Path
from random import randint

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorAmbi:
    dac_base_path: Path
    clip_base_path: Path
    rotation_base_path: Path = None
    energy_map_path: Path = None
    seconds_to_use: int = 5
    dac_pad_token_id: int = 1024
    dac_num_codebooks: int = 9
    label_pad_token_id: int = -100
    dac_frame_rate: int = 86
    clip_frame_rate: int = 4
    ambi_path = "csv/yt_ambigen/ambi.json"

    def __post_init__(self):
        if isinstance(self.dac_base_path, str):
            self.dac_base_path = Path(self.dac_base_path)
        if isinstance(self.clip_base_path, str):
            self.clip_base_path = Path(self.clip_base_path)
        if isinstance(self.energy_map_path, str):
            self.energy_map_path = Path(self.energy_map_path)
        if isinstance(self.rotation_base_path, str):
            self.rotation_base_path = Path(self.rotation_base_path)
        else:
            self.rotation_base_path = None

        with open(self.ambi_path) as f:
            self.ambi_data = json.load(f)

    def roty(self, ang):
        return np.array(
            [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
        )

    def rotx(self, ang):
        return np.array(
            [[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]
        )

    def prepare_ambi(self, file_path, augment=False):
        key = file_path.replace(".npy", "")
        elevation, azimuth = self.ambi_data[key]
        if augment:
            azimuth = azimuth + 45
            if azimuth > 179:
                azimuth = azimuth - 180
        direction = (
            self.roty((2 * azimuth - 179) * np.pi / 180)
            @ self.rotx((89 - 2 * elevation) * np.pi / 180)
            @ np.array([0.0, 0.0, 1.0])
        )
        direction = np.float32(direction)

        return direction

    def prepare_dac(self, dac: np.ndarray) -> np.ndarray:
        """
        Input: DAC codes of shape [s, n]
        Output: Decoder input ids of shape [2s+1, n]
        """
        # DAC -> Pad & Shift
        decoder_input_ids = []
        residual_codebooks = tuple(range(1, self.dac_num_codebooks))

        for idx in range(dac.shape[1]):
            decoder_input_id = np.full(
                dac.shape[0] * 2 + 1, self.dac_pad_token_id, dtype=dac.dtype
            )

            if idx == 0:  # W_p
                decoder_input_id[::2][:-1] = dac[:, idx]
            elif (
                idx in residual_codebooks or idx % self.dac_num_codebooks == 0
            ):  # W_r and S_p
                decoder_input_id[1::2] = dac[:, idx]
            else:  # S_r
                decoder_input_id[2::2] = dac[:, idx]
            decoder_input_ids.append(decoder_input_id)

        decoder_input_ids = np.stack(decoder_input_ids, axis=1)

        return decoder_input_ids

    def prepare_labels(self, decoder_input_ids: np.ndarray) -> np.ndarray:
        """
        Input: Decoder input ids of shape [2s+1, n]
        Output: Labels of shape [2s+1, n]
        """
        # Labels -> Add pad token for last time step
        labels = decoder_input_ids.copy()
        labels[labels == self.dac_pad_token_id] = self.label_pad_token_id

        return labels

    def preprocess_train(self, example):
        output = {
            "inputs_embeds": [],
            "decoder_input_ids": [],
            "labels": [],
            "direction": [],
        }
        if self.energy_map_path is not None:
            output["energy_map"] = []

        for file_path in example["file_path"]:
            try:
                rnd = random.random()
                if self.rotation_base_path is not None and rnd < 0.5:
                    dac = np.load(self.rotation_base_path / file_path)
                    augment = True
                else:
                    dac = np.load(self.dac_base_path / file_path)
                    augment = False

                dac = np.transpose(dac, axes=(1, 0))
                dac = dac[:, : self.dac_num_codebooks * 4]
                clip_embedding = np.load(self.clip_base_path / file_path)
                if self.energy_map_path is not None:
                    energy_map = np.load(self.energy_map_path / file_path)
                else:
                    energy_map = None
                direction = self.prepare_ambi(file_path, augment=augment)
            except Exception as e:
                print(e)
                raise ValueError(f"Error processing file {file_path}")

            try:
                max_second = dac.shape[0] // self.dac_frame_rate
                if max_second > self.seconds_to_use:
                    max_second = min(
                        max_second, clip_embedding.shape[0] // self.clip_frame_rate
                    )
                    start_second = randint(0, max_second - self.seconds_to_use)
                    clip_embedding = clip_embedding[
                        start_second
                        * self.clip_frame_rate : (start_second + self.seconds_to_use)
                        * self.clip_frame_rate
                    ]
                    if energy_map is not None:
                        energy_map = energy_map[
                            start_second
                            * self.clip_frame_rate : (
                                start_second + self.seconds_to_use
                            )
                            * self.clip_frame_rate
                        ]
                    dac = dac[
                        start_second
                        * self.dac_frame_rate : (start_second + self.seconds_to_use)
                        * self.dac_frame_rate
                    ]
                else:
                    clip_embedding = clip_embedding[
                        : self.seconds_to_use * self.clip_frame_rate
                    ]
                    if energy_map is not None:
                        energy_map = energy_map[
                            : self.seconds_to_use * self.clip_frame_rate
                        ]
                    dac_pad = self.seconds_to_use * self.dac_frame_rate - dac.shape[0]
                    if dac_pad > 0:
                        dac = np.pad(
                            dac,
                            ((0, dac_pad), (0, 0)),
                            constant_values=self.dac_pad_token_id,
                        )
                    else:
                        dac = dac[: self.seconds_to_use * self.dac_frame_rate]
            except:
                raise ValueError(f"Error processing example: {file_path}")

            if dac.shape[0] != self.dac_frame_rate * self.seconds_to_use:
                raise ValueError(f"Error processing example {file_path}")

            inputs_embeds = clip_embedding
            decoder_input_ids = self.prepare_dac(dac)
            labels = self.prepare_labels(decoder_input_ids)

            output["inputs_embeds"].append(torch.tensor(inputs_embeds))
            output["decoder_input_ids"].append(torch.tensor(decoder_input_ids[:-1]))
            output["labels"].append(torch.tensor(labels))
            output["direction"].append(torch.from_numpy(direction))
            if energy_map is not None:
                output["energy_map"].append(torch.tensor(energy_map))

        return output

    def preprocess_eval(self, example):
        try:
            dac = np.load(self.dac_base_path / example["file_path"].strip())
            dac = np.transpose(dac, axes=(1, 0))
            dac = dac[:, : self.dac_num_codebooks * 4]
            clip_embedding = np.load(self.clip_base_path / example["file_path"].strip())
            direction = self.prepare_ambi(example["file_path"].strip(), augment=False)
            if self.energy_map_path is not None:
                energy_map = np.load(
                    self.energy_map_path / example["file_path"].strip()
                )
            else:
                energy_map = None
        except:
            raise ValueError(f"Error processing example: {example['file_path']}")

        # Temporary truncate longer sequences
        max_second = dac.shape[0] // self.dac_frame_rate
        if max_second > self.seconds_to_use:
            start_second = 0
            clip_embedding = clip_embedding[
                start_second
                * self.clip_frame_rate : (start_second + self.seconds_to_use)
                * self.clip_frame_rate
            ]
            dac = dac[
                start_second
                * self.dac_frame_rate : (start_second + self.seconds_to_use)
                * self.dac_frame_rate
            ]
            if energy_map is not None:
                energy_map = energy_map[
                    start_second
                    * self.clip_frame_rate : (start_second + self.seconds_to_use)
                    * self.clip_frame_rate
                ]
        else:
            clip_embedding = clip_embedding[
                : self.seconds_to_use * self.clip_frame_rate
            ]
            if energy_map is not None:
                energy_map = energy_map[: self.seconds_to_use * self.clip_frame_rate]
            dac_pad = self.seconds_to_use * self.dac_frame_rate - dac.shape[0]
            if dac_pad > 0:
                dac = np.pad(
                    dac, ((0, dac_pad), (0, 0)), constant_values=self.dac_pad_token_id
                )
            else:
                dac = dac[: self.seconds_to_use * self.dac_frame_rate]

        inputs_embeds = clip_embedding
        decoder_input_ids = self.prepare_dac(dac)
        labels = self.prepare_labels(decoder_input_ids)

        return_dict = {
            "inputs_embeds": inputs_embeds,
            "decoder_input_ids": decoder_input_ids[:-1],
            "labels": labels,
            "direction": direction,
        }

        if energy_map is not None:
            return_dict["energy_map"] = energy_map

        return return_dict
