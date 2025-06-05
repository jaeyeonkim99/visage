# Instruction for preparing dataset

## Environment

```bash
conda create -y -n visagedl -c conda-forge python=3.8 ffmpeg=5.1.2
conda activate visagedl
pip install ffmpeg-python tqdm yt-dlp numpy scipy opencv-python
```

After setting up your environment, specify your desired output path in `prepare_dataset.py`.


## Download raw videos

```bash
python prepare_dataset.py crawl
```

Please note that downloading FOA with yt-dlp could be [unstable](https://github.com/yt-dlp/yt-dlp/issues/12543).


## Crop 5-second clips

```bash
python prepare_dataset.py clip
```


## Extract field-of-view clips

```bash
python prepare_dataset.py nfov
```