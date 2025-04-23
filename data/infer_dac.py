import argparse
import io
import math
import os

import dac
import ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from audiotools import AudioSignal
from tqdm import tqdm


def read_audio_from_mp4(file_path):
    # Run ffmpeg to extract audio in wav format and pipe it to stdout
    process = (
        ffmpeg.input(file_path)
        .output("pipe:", format="wav")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    # Read the output and load it into a bytes buffer
    out, _ = process.communicate()
    audio_buffer = io.BytesIO(out)

    # Use soundfile to read from the buffer
    data, samplerate = sf.read(audio_buffer)
    return data, samplerate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv",
        "-dc",
        type=str,
        default="yt_ambigen.csv",
        help="Path of the CSV file containing input video paths",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        help="Path of the original .opus files",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path to save dac .npy files",
    )
    parser.add_argument(
        "--n_quantizers", "-n", type=int, default=9, help="Number of quantizers to use"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=20,
        help="Batch size to run the inference",
    )
    parser.add_argument(
        "--dac_pad_value",
        "-p",
        default=1024,
        help="The value to pad dac codes for unexisting channels",
    )
    parser.add_argument(
        "--rotate",
        "-r",
        action="store_true",
        help="Rotate the ambisonics or not",
    )
    parser.add_argument(
        "--dac_sample_rate",
        "-dsr",
        type=int,
        default=44100,  # Or 16000 if you want
        help="Sample rate of the DAC model",
    )
    parser.add_argument(
        "--audio_extension",
        "-ae",
        type=str,
        default=".opus",
        help="Extension of the audio files",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dac_sample_rate == 44100:
        dac_frame_rate = 512
        model_path = dac.utils.download(model_type="44khz")
    elif args.dac_sample_rate == 16000:
        dac_frame_rate = 320
        model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path)
    model.eval()
    model = model.to(device)

    clips = pd.read_csv(args.data_csv)["id"].tolist()
    wavs = [f"{args.data_path}/{clip}{args.audio_extension}" for clip in clips]
    print_code = True

    with torch.no_grad():
        for idx in tqdm(range(0, len(wavs), args.batch_size)):
            if idx + args.batch_size >= len(wavs):
                batch = wavs[idx:]
            else:
                batch = wavs[idx : idx + args.batch_size]

            signals = []
            lengths = []
            save_paths = []
            for wav in batch:
                save_path = wav.replace(args.data_path, args.save_path).replace(
                    args.audio_extension, ".npy"
                )
                if os.path.exists(save_path):
                    continue
                audio, sr = sf.read(wav)
                audio = np.transpose(audio, (-1, -2))

                if args.rotate:
                    audio[1], audio[3] = -audio[3].copy(), audio[1].copy()

                save_paths.append(save_path)
                for channel_idx, channel in enumerate(audio):
                    signal = AudioSignal(channel, sr).to(device)
                    signal.resample(args.dac_sample_rate)
                    signals.append(signal)
                lengths.append(math.ceil(signal.shape[-1] / dac_frame_rate))

            if len(save_paths) == 0:
                continue

            batched_signal = AudioSignal.batch(signals, pad_signals=True)
            x = model.preprocess(batched_signal.audio_data, batched_signal.sample_rate)
            _, codes, _, _, _ = model.encode(x, n_quantizers=args.n_quantizers)
            codes = codes[:, : args.n_quantizers]

            # Codes: [4*batch, n_quantizer, time]
            codes = codes.cpu().numpy()
            for idx, save_path in enumerate(save_paths):
                code = codes[idx * 4 : idx * 4 + 4]
                code = code.reshape(-1, code.shape[-1])
                if print_code:
                    print(code.shape)
                    print_code = False

                # Unpad the codes
                code = code[:, : lengths[idx]]

                # Fill pad value for unexisting channels
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, code)
