# ViSAGe: Video-to-Spatial Audio Generation (ICLR 2025)

Official implementation of "ViSAGe: Video-to-Spatial AUdio Generation" (ICLR 2025)

[[Paper]](https://openreview.net/pdf?id=8bF1Vaj9tm) [[Project Page]](https://jaeyeonkim99.github.io/visage/)  [[Poster]](assets/visage_poster.pdf)

**Abstract**: Spatial audio is essential for enhancing the immersiveness of audio-visual experiences, yet its production typically demands complex recording systems and specialized expertise. In this work, we address a novel problem of generating firstorder ambisonics, a widely used spatial audio format, directly from silent videos. To support this task, we introduce YT-Ambigen, a dataset comprising 102K 5-second YouTube video clips paired with corresponding first-order ambisonics. We also propose new evaluation metrics to assess the spatial aspect of generated audio based on audio energy maps and saliency metrics. Furthermore, we present Videoto-Spatial Audio Generation (ViSAGe), an end-to-end framework that generates first-order ambisonics from silent video frames by leveraging CLIP visual features, autoregressive neural audio codec modeling with both directional and visual guidance. Experimental results demonstrate that ViSAGe produces plausible and
coherent first-order ambisonics, outperforming two-stage approaches consisting of video-to-audio generation and audio spatialization. Qualitative examples further
illustrate that ViSAGe generates temporally aligned high-quality spatial audio that adapts to viewpoint changes. Project page: [https://jaeyeonkim99.github.io/visage](https://jaeyeonkim99.github.io/visage)

![Overall](assets/keyidea.png)

## [YT-Ambigen](utils/README.md)

- Data splits are provided in `csv/yt_ambigen/{train, valid, test}.csv`.
- Camera direction for each clip is given in `csv/yt_ambigen/ambi.json`, in `(elevation, azimuth)` format.
  - `elevation`: Values range from 0 to 90, representing the angle (in degrees) from the vertical (z-axis). Each unit corresponds to 2 degrees.
  - `azimuth`: Values range from 0 to 180, representing the anti-clockwise angle from the y-axis (back side). Each unit corresponds to 2 degrees.
- ~~Scripts to help download and prepare the dataset will be released soon.~~
- Scripts for data preparation are released! Please refer to [`utils/README.md`](utils/README.md) for details.
- We plan to release 3x larger version of YT-Ambigen with 100% human validated test samples. Stay tuned for further updates.

## Environments

- Install dependencies using `pip install -r requirements.txt`
- If your GPU supports Flash Attention 2, we recommend installing [flash-attn](https://github.com/Dao-AILab/flash-attention) for training.
- Install [audioldm_eval](https://github.com/haoheliu/audioldm_eval) for evaluation. Use branch [passt_replace_panns](https://github.com/haoheliu/audioldm_eval/tree/passt_replace_panns).
- Install [openai/CLIP](https://github.com/openai/CLIP) to preprocess data.

## Training

### 1. Preprocess dataset

- Prepare DAC and CLIP features using scripts in `data/`. Detailed instructions are in [`data/README.md`](data/README.md).

### 2. Setup training and model configuration

- Modify `cfg/visage_vggsound.yaml` or `cfg/visage_ytambigen.yaml` for training configurations:

```yaml
output_dir: /output  # Path for checkpoints and logs

dac_base_path: /data/yt_ambigen/dac  # DAC features path
rotation_base_path: /data/yt_ambigen/dac_rotated  # Rotated DAC features path
clip_base_path: /data/yt_ambigen/clip  # CLIP features path
energy_map_path: /data/yt_ambigen/energy_map  # Energy map features path

model_path: null  # Pretrained checkpoint path (optional)
```

- Modify `cfg/config.json` for model configurations if necessary.

### 3. Run training

```bash
# Setup accelerate
accelerate config

# Execute training script
sh train_ambi.sh
```

## Evaluation

- Prepare test audio files and ground-truth energy maps as detailed in [`data/README.md`](data/README.md).
- Adjust environment variables in `evaluate_ambi.sh` accordingly, then run:

```bash
sh evaluate_ambi.sh
```

## Pretrained Checkpoints

- Pretrained checkpoints are available at [huggingface](https://huggingface.co/jaeyeonkim99/visage/tree/main). 
  
## Citation

```bibtex
@inproceedings{kimvisage,
  title={ViSAGe: Video-to-Spatial Audio Generation},
  author={Kim, Jaeyeon and Yun, Heeseung and Kim, Gunhee},
  booktitle={ICLR},
  year={2025}
}
```
