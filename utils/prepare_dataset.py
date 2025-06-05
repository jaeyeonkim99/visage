import sys
import csv
import json
import time
import random
import subprocess as sp
import multiprocessing as mp
from pathlib import Path

import cv2
import ffmpeg
import yt_dlp
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d


OUT_PATH = Path("./your/output/path")  # Change this to your desired output path


def sec2time(sec):
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def rotx(ang):
    return np.array([
        [1, 0, 0],
        [0, np.cos(ang), -np.sin(ang)],
        [0, np.sin(ang), np.cos(ang)]
    ])

def roty(ang):
    return np.array([
        [np.cos(ang), 0, np.sin(ang)],
        [0, 1, 0],
        [-np.sin(ang), 0, np.cos(ang)]
    ])

class DlLogger(object):
    def debug(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        pass
    def info(self, msg):
        pass

YTDL_OPTS = {
    "outtmpl": str(OUT_PATH / "raw" / "%(id)s.%(ext)s"),
    "logger": DlLogger(),
    "quiet": True,
    "no_warnings": True,
    "extractor_args": {"youtube": {"player_client": ["default", "android_vr"]}},
}

def select_format(ctx):
    formats = ctx.get("formats")
    a_fmt = [
        f for f in formats
        if f.get("acodec", "none") != "none" and
        f.get("vcodec", "none") == "none" and
        f.get("audio_channels", 0) > 2 and
        f.get("format_note", "").find("ambi") > -1
    ]
    v_fmt = [
        f for f in formats
        if f.get("acodec", "none") == "none" and
        f.get("vcodec", "none") != "none" and
        f.get("ext", "none") == "mp4" and
        f.get("height", 0) >= 480
    ]
    if len(a_fmt) == 0:
        return None
    elif len(v_fmt) == 0:
        return None
    else:
        return {
            "video": v_fmt[0],
            "audio": a_fmt[0],
            "title": ctx["title"]
        }
    

def crawl(id):
    time.sleep(random.randint(1,5))
    if (OUT_PATH / "raw" / f"{id}.mp4").exists():
        if (OUT_PATH / "raw" / f"{id}.webm").exists():
            return
    url = f"https://www.youtube.com/watch?v={id}"
    try:
        with yt_dlp.YoutubeDL(YTDL_OPTS) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(str(e).lower())
        return
    curr_opts = {k:v for k,v in YTDL_OPTS.items()}
    stats = select_format(info)
    if stats is None:
        return
    curr_opts["format"] = f"{stats['audio']['format_id']},"
    curr_opts["format"] += f"{stats['video']['format_id']}"
    try:
        with yt_dlp.YoutubeDL(curr_opts) as ydl:
            _ = ydl.download(url)
        json.dump(
            stats,
            open(OUT_PATH / "raw" / f"{id}.json", 'w'),
            indent=4
        )
    except:
        pass


def clip(video_id, sec_id):
    v_meta = ffmpeg.probe(str(OUT_PATH / "raw" / f"{video_id}.mp4"))
    a_meta = ffmpeg.probe(str(OUT_PATH / "raw" / f"{video_id}.webm"))

    if not (OUT_PATH / "video" / f"{video_id}_{sec_id}.mp4").exists():
        cmd = "ffmpeg -loglevel warning -y"
        cmd += f" -ss {sec2time(sec_id)} -to {sec2time(sec_id+5)}"
        cmd += " -i " + str(OUT_PATH / "raw" / f"{video_id}.mp4")
        cmd += f" -r 4 -s 896x448 "
        cmd += str(OUT_PATH / "video" / f"{video_id}_{sec_id}.mp4")
        try:
            _ = sp.run(cmd, shell=True, capture_output=True)
        except sp.CalledProcessError as e:
            print(e.output)
            (OUT_PATH / "video" / f"{video_id}_{sec_id}.mp4").unlink()
    
    if not (OUT_PATH / "audio" / f"{video_id}_{sec_id}.opus").exists():
        cmd = f"ffmpeg -loglevel warning -y"
        cmd += " -i " + str(OUT_PATH / "raw" / f"{video_id}.webm")
        cmd += f" -ss {sec2time(sec_id)} -to {sec2time(sec_id+5)}"
        cmd += f" -c:a copy "
        cmd += str(OUT_PATH / "audio" / f"{video_id}_{sec_id}.opus")
        try:
            _ = sp.run(cmd, shell=True, capture_output=True)
        except sp.CalledProcessError as e:
            print(e.output)
            (OUT_PATH / "audio" / f"{video_id}_{sec_id}.opus").unlink()


def project(coord, pano_img):
    view = [coord[1] * 2 - 179, -(coord[0] * 2 - 89)]
    view = [view[0] * np.pi / 180, view[1] * np.pi / 180]
    vfov = 0.5 * np.pi
    in_w, in_h = 896, 448
    out_w, out_h = 224, 224
    
    topLeft = np.array([
        -np.tan(vfov / 2) * (out_w / out_h),
        -np.tan(vfov / 2),
        1
    ])
    uv = np.array([
        -2 * topLeft[0] / out_w,
        -2 * topLeft[1] / out_h,
        0
    ])
    res_acos, res_atan = 2 * in_w, 2 * in_h
    step_acos = np.pi / res_acos
    step_atan = np.pi / res_atan
    lookup_acos = -np.cos(np.linspace(0, np.pi, 2 * in_w + 1))
    lookup_atan = np.tan(np.linspace(-np.pi / 2, np.pi / 2, 2 * in_h + 1))
    lookup_atan[0] = np.tan(step_atan / 2 - np.pi / 2)
    lookup_atan[-1] = np.tan(-step_atan / 2 + np.pi / 2)

    X, Y = np.meshgrid(range(out_w), range(out_h))
    X, Y = X.flatten(), Y.flatten()
    yaw, pitch = view[0], view[1]
    points = np.concatenate(
        (
            topLeft[0] + uv[0] * np.expand_dims(X, axis=0),
            topLeft[1] + uv[1] * np.expand_dims(Y, axis=0),
            np.ones((1, X.shape[0]))
        ),
        axis=0
    )
    moved_points = roty(yaw) @ rotx(pitch) @ points
    x_points = moved_points[0, :]
    y_points = moved_points[1, :]
    z_points = moved_points[2, :]

    nxz = np.sqrt(x_points**2 + z_points**2)
    phi = np.zeros(X.shape[0])
    theta = np.zeros(X.shape[0])

    ind = nxz < 10e-10
    phi[ind & (y_points > 0)] = np.pi / 2
    phi[ind & (y_points <= 0)] = -np.pi / 2

    ind = np.logical_not(ind)
    phi_interp = interp1d(
        lookup_atan,
        np.arange(0, res_atan + 1),
        'linear',
        fill_value="extrapolate"
    )
    phi[ind] = phi_interp(y_points[ind] / nxz[ind]) * step_atan - np.pi / 2
    theta_interp = interp1d(
        lookup_acos,
        np.arange(0, res_acos+1),
        'linear'
    )
    theta[ind] = theta_interp(-z_points[ind] / nxz[ind]) * step_acos
    theta[ind & (x_points < 0)] = -theta[ind & (x_points < 0)]

    inX = (theta / np.pi) * (in_w / 2) + (in_w / 2) + 1
    inY = (phi / (np.pi / 2)) * (in_h / 2) + (in_h / 2) + 1

    inX[inX < 1] = 1
    inX[inX >= in_w - 1] = in_w - 1
    inY[inY < 1] = 1
    inY[inY >= in_h-1] = in_h - 1

    out = np.zeros((out_h, out_w, pano_img.shape[2]), pano_img.dtype)
    inX = inX.reshape(out_w, out_h).astype('float32')
    inY = inY.reshape(out_w, out_h).astype('float32')
    for c in range(pano_img.shape[2]):
        out[:, :, c] = cv2.remap(pano_img[:, :, c], inX, inY, cv2.INTER_LINEAR)
    return out


def nfov(clip_id, coord):
    in_path = OUT_PATH / "video" / f"{clip_id}.mp4"
    out_path = OUT_PATH / "nfov" / f"{clip_id}.mp4"
    if out_path.exists():
        return
    metadata = ffmpeg.probe(str(in_path))["streams"][0]
    in_w, in_h = int(metadata["width"]), int(metadata["height"])
    pano_vid, _ = (
        ffmpeg
        .input(str(in_path))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=True)
    )
    pano_vid = np.frombuffer(pano_vid, np.uint8).reshape([-1, in_h, in_w, 3])
    out_vid = list()
    for pano_img in pano_vid:
        out_vid.append(project(coord, pano_img))
    proc = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4, (224, 224)
    )
    for out_img in out_vid:
        proc.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    proc.release()


if __name__ == "__main__":
    command = sys.argv[-1]
    if command == "crawl":
        (OUT_PATH / "raw").mkdir(exist_ok=True, parents=True)
        video_list = set()
        for split in ["train", "valid", "test"]:
            split_path = Path(f"../csv/yt_ambigen/{split}.csv")
            video_list.update([
                data[0][:11] for data in 
                list(csv.reader(open(split_path, 'r')))[1:]
            ])
        video_list = list(video_list)[:5]
        for video_id in tqdm(video_list):
            crawl(video_id)
    elif command == "clip":
        (OUT_PATH / "video").mkdir(exist_ok=True, parents=True)
        (OUT_PATH / "audio").mkdir(exist_ok=True, parents=True)
        video_list = set([p.stem for p in (OUT_PATH / "raw").glob("*.mp4")])
        segment_list = list()
        for split in ["train", "valid", "test"]:
            split_path = Path(f"../csv/yt_ambigen/{split}.csv")
            for file_path in list(csv.reader(open(split_path, 'r')))[1:]:
                video_id = file_path[0][:11]
                sec_id = int(file_path[0][12:].split(".")[0])
                if video_id not in video_list:
                    continue
                segment_list.append((video_id, sec_id))
        for video_id, sec_id in tqdm(segment_list):
            clip(video_id, sec_id)
    elif command == "nfov":
        (OUT_PATH / "nfov").mkdir(exist_ok=True, parents=True)
        video_list = set([p.stem for p in (OUT_PATH / "video").glob("*.mp4")])
        ambi_dict = json.load(open("../csv/yt_ambigen/ambi.json", 'r'))
        for video_id in tqdm(video_list):
            nfov(video_id, ambi_dict[video_id])
    else:
        raise NotImplementedError(f"Unknown command: {command}")
        