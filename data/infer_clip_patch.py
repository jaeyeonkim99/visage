import argparse
import os

import clip
import ffmpeg
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
        help="Path of input videos",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path to save clip features",
    )
    parser.add_argument(
        "--video_extension",
        "-v",
        type=str,
        default=".mp4",
        help="Extension of the video files",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=40,
        help="Batch size to run the inference",
    )
    parser.add_argument(
        "--fps", "-f", type=int, default=4, help="Frame per second to get clip features"
    )
    args = parser.parse_args()

    # Initiate CLIP model
    print("Running for ", args.data_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device="cuda")
    print(f"CLIP model loaded successfully")

    # Prepare video paths
    df = pd.read_csv(args.data_csv)
    clips = df["id"].tolist()
    videos = [
        f"{args.data_path}/{video_clip}{args.video_extension}" for video_clip in clips
    ]
    input_resolution = 224
    print_shape = True

    with torch.no_grad():
        for idx in tqdm(
            range(0, len(videos), args.batch_size),
            dynamic_ncols=True,
        ):
            # for idx in iterator:
            if idx + args.batch_size >= len(videos):
                batch = videos[idx:]
            else:
                batch = videos[idx : idx + args.batch_size]

            save_paths = []
            lengths = []
            video_batch = []
            for video_file in batch:
                save_path = video_file.replace(args.data_path, args.save_path).replace(
                    args.video_extension, ".npy"
                )
                if os.path.exists(save_path):
                    continue
                save_paths.append(save_path)

                # Read video
                probe = ffmpeg.probe(video_file)
                video_stream = next(
                    (
                        stream
                        for stream in probe["streams"]
                        if stream["codec_type"] == "video"
                    ),
                    None,
                )
                orig_width = int(video_stream["width"])
                orig_height = int(video_stream["height"])
                width = (
                    input_resolution
                    if orig_width < orig_height
                    else input_resolution * orig_width // orig_height
                )
                height = (
                    input_resolution
                    if orig_width >= orig_height
                    else input_resolution * orig_height // orig_width
                )
                x = (width - input_resolution) // 2
                y = (height - input_resolution) // 2
                vid, _ = (
                    ffmpeg.input(video_file)
                    .filter("fps", fps=args.fps)
                    .filter("scale", width, height)
                    .crop(x, y, input_resolution, input_resolution)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                    .run(capture_stdout=True, quiet=True)
                )

                # [T, 224, 224, 3]
                video_frames = np.frombuffer(vid, np.uint8).reshape(
                    [-1, input_resolution, input_resolution, 3]
                )

                # [T, 3, 224, 224]
                video_frames = (
                    torch.from_numpy(video_frames.astype("float32") / 255)
                    .permute(0, 3, 1, 2)
                    .to(device)
                )
                # Preprocess video frames
                video_frames = preprocess.transforms[-1](video_frames)

                # Each video input: [T, 3, 224, 224]
                video_batch.append(video_frames)
                lengths.append(video_frames.shape[0])

            # Create batch and run the model
            # Model input: [B & T , 3, 224, 224]
            if len(save_paths) > 0:
                video_batch = torch.cat(video_batch, dim=0)

                # Video feat: [B & T, 50, 768]
                video_feat = model.encode_image(video_batch)
                video_feat = video_feat.cpu().numpy()

                # Save clip features
                length_idx = 0
                for idx, save_path in enumerate(save_paths):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    clip_feats = video_feat[
                        length_idx : length_idx + lengths[idx]
                    ]  # [T, 50, 768]
                    if clip_feats.shape[0] == 0:
                        print(
                            f"Error processing time axis for {save_path}: {clip_feats.shape} "
                        )
                    if print_shape:
                        print(clip_feats.shape)
                        print_shape = False
                    np.save(save_path, clip_feats)
                    length_idx += lengths[idx]
