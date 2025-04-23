import argparse
import os
import pdb

import dac
import torch
import torchaudio
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from tqdm.auto import tqdm

from data.preprocess_ambi import PreprocessorAmbi
from modeling.visage import VisageForAmbisonicsGeneration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save_path", type=str, help="Path to save the generated audio"
    )
    parser.add_argument(
        "-c", "--ckpt", type=str, required=False, help="Model checkpoint path"
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=-1, required=False, help="Epoch to use"
    )
    parser.add_argument(
        "-tc",
        "--test_csv",
        type=str,
        default="csv/ijcv_all/test.csv",
        help="CSV file with list of test files for evaluation",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to run the inference",
    )

    parser.add_argument("-i", "--index", type=int, default=-1, help="index of csv file")

    args = parser.parse_args()

    # Load model to evaluate
    if args.epoch > -1:
        ckpt_path = os.path.join(args.ckpt, "checkpoints", f"epoch_{args.epoch}")
    else:
        ckpt_path = args.ckpt

    model = VisageForAmbisonicsGeneration.from_pretrained(ckpt_path)
    print(f"> Loaded model from {ckpt_path}")

    if args.index > -1:
        args.test_csv = f"{args.test_csv.replace('.csv', '')}/{args.index}.csv"
        print("> Running for index", args.index)

    model = model.cuda()
    model.eval()
    config = model.config

    # Load args of the model
    args_path = os.path.join(args.ckpt, "args.yaml")
    if os.path.exists(args_path):
        model_args = OmegaConf.load(args_path)
    else:
        model_args = OmegaConf.load("cfg/visage_ytambigen.yaml")

    # Prepare preprocessor
    preprocessor = PreprocessorAmbi(
        dac_base_path=model_args.dac_base_path,
        clip_base_path=model_args.clip_base_path,
        energy_map_path=model_args.energy_map_path,
        rotation_base_path=None,
        seconds_to_use=model_args.seconds_to_use,
        dac_pad_token_id=config.dac_pad_token_id,
        dac_num_codebooks=config.num_rvq,
        dac_frame_rate=config.dac_frame_rate,
        clip_frame_rate=config.clip_frame_rate,
        label_pad_token_id=-100,
    )

    # Prepare DAC
    dac_sample_rate = config.dac_sample_rate
    if dac_sample_rate == 16000:
        dac_model_path = dac.utils.download(model_type="16khz")
        resample_transform = None
    else:
        dac_model_path = dac.utils.download(model_type="44khz")
        resample_transform = Resample(orig_freq=44100, new_freq=16000).cuda()

    dac_model = dac.DAC.load(dac_model_path)
    dac_model = dac_model.cuda()
    dac_model.eval()

    # Load test set
    data_files = {"test": args.test_csv}
    raw_datasets = load_dataset("csv", data_files=data_files)
    test_dataset = raw_datasets["test"].map(
        preprocessor.preprocess_eval,
        num_proc=model_args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc="Praparing test dataset",
    )
    test_dataset.set_format(
        "pt", columns=["file_path", "inputs_embeds", "direction", "energy_map"]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Generate and save the wav file
    print("> Generating wav files...")
    test_iterator = tqdm(test_dataloader, colour="YELLOW", dynamic_ncols=True)
    model.guidance_debug = True
    with torch.no_grad():
        for step, batch in enumerate(test_iterator):
            if config.classifier_free_guidance:
                generated_codes = model.generate_cfg(
                    batch["inputs_embeds"].cuda(),
                    direction=batch["direction"].cuda(),
                    energy_map=batch["energy_map"].cuda(),
                    do_sample=True,
                    top_k=256,
                    guidance_scale=2.5,
                )
            else:
                generated_codes = model.generate(
                    batch["inputs_embeds"].cuda(), do_sample=True, top_k=256
                )
            generated_codes = generated_codes.transpose(-1, -2)
            # Get
            audios = []
            code_len = generated_codes.shape[1]
            for i in range(4):
                channel = generated_codes[
                    :,
                    i * config.num_rvq : (i * config.num_rvq + config.num_rvq),
                ]
                z = dac_model.quantizer.from_codes(channel)[0]
                audio = dac_model.decode(z).detach()
                audios.append(audio)
            audios = torch.cat(audios, dim=1)

            for idx, file_path in enumerate(batch["file_path"]):
                out_path = os.path.join(args.save_path, file_path).replace(
                    ".npy", ".wav"
                )
                save_audio = audios[idx]
                if resample_transform:
                    save_audio = resample_transform(save_audio)
                save_audio = save_audio.cpu()
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torchaudio.save(out_path, save_audio, 16000)
