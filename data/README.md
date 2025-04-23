# Data Preprocessing for ViSAGe

- You can see detailed usage for each script by adding the `-h` flag.
- Files are processed based on video IDs provided in the CSV file (`yt_ambigen.csv`).  
  You may split the CSV file to parallelize feature extraction across multiple processes.

## Preparing DAC and CLIP

```bash
# Extract DAC features
python infer_dac.py --data_csv yt_ambigen.csv --data_path AMBISONICS_PATH --save_path DAC_SAVE_PATH  # Expected shape: (36, T)

# Extract rotated DAC features for rotation augmentation
python infer_dac.py --data_csv yt_ambigen.csv --data_path AMBISONICS_PATH --save_path DAC_ROTATE_SAVE_PATH --rotate  # Expected shape: (36, T)

# Extract CLIP features
python infer_clip.py --data_csv yt_ambigen.csv --data_path VIDEO_PATH --save_path CLIP_SAVE_PATH  # Expected shape: (T, 512)
```

## Preparing Energy Map

- To extract features for the energy map, you must first modify the installed `clip` library.
- Comment out lines 235–238 in [`clip/model.py`](https://github.com/openai/CLIP/blob/main/clip/model.py):

```python
def forward(self, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # x = self.ln_post(x[:, 0, :])

    # if self.proj is not None:
    #     x = x @ self.proj

    return x
```

- After making the modification, run the following commands:

```bash
# Extract patch-based CLIP features
python infer_clip_patch.py --data_csv yt_ambigen.csv --data_path VIDEO_PATH --save_path CLIP_PATCH_SAVE_PATH  # Expected shape: (T, 50, 768)

# Compute patch-wise energy map
python infer_clip_energy.py --clip_patch_path CLIP_PATCH_SAVE_PATH --save_path ENERGY_MAP_SAVE_PATH  # Expected shape: (T, 49)
```

## Preparing Evaluation
- We provide scripts to prepare audio files for evaluation.

- After preparing the test audio files, generate the energy map for ground-truth ambisonics required for spatial evaluation:

```bash
# Prepare test audio files
python prepare_test.py --data_path AMBISONICS_PATH --save_path TEST_AUDIO_PATH

# Generate ground-truth energy map
python prepare_gt_energy_map.py --gt_audio_path TEST_AUDIO_PATH --save_path GT_ENERGY_MAP_PATH
```

