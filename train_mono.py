#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import os
import sys
from datetime import datetime

import dac
import datasets
import pandas as pd
import torch
import torch.nn as nn
import transformers
from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_inverse_sqrt_schedule, get_scheduler

from data.preprocess import Preprocessor
from modeling.visage_mono import VisageConfig, VisageForConditionalGeneration
from utils import count_parameters, count_trainable_parameters

logger = get_logger(__name__)


def main():
    # Load Configuration
    cfg_path = sys.argv[1]
    args = OmegaConf.load(cfg_path)

    # Initialize Logging
    accelerator_log_kwargs = {}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    dataloader_config = DataLoaderConfiguration(split_batches=args.split_batches)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs],
        **accelerator_log_kwargs,
    )
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            OmegaConf.save(args, f)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARNING,
    )
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train_log.txt"))
    logger.logger.addHandler(file_handler)
    logger.warning(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets
    data_files = {}
    data_files_eval = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files_eval["validation"] = args.validation_file
    if args.demo_file is not None:
        data_files_eval["demo"] = args.demo_file

    # Prepare metadata for demo files
    if accelerator.is_main_process and hasattr(args, "demo_metadata"):
        demo_df = pd.read_csv(args.demo_metadata)
        demo_metadata = {
            item["file_path"]: item["category"] for _, item in demo_df.iterrows()
        }

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    raw_datasets_eval = load_dataset(extension, data_files=data_files_eval)

    # Load config and model
    if hasattr(args, "model_path") and args.model_path is not None:
        config = VisageConfig.from_pretrained(
            os.path.join(args.model_path), "config.json"
        )
        config._attn_implementation = "flash_attention_2"
        model = VisageForConditionalGeneration.from_pretrained(
            args.model_path, config=config, ignore_mismatched_sizes=False
        )
        logger.warning(f"Loading pretrained model from {args.model_path}")
    else:
        config = VisageConfig.from_pretrained(args.model_cfg)
        model = VisageForConditionalGeneration(config=config)

    if hasattr(args, "freeze_decoder") and args.freeze_decoder is True:
        model.freeze_decoder()

    logger.warning(
        f"Total number of parameters: {count_parameters(model)/1000000:.2f}M"
    )
    logger.warning(
        f"Total number of trainable parameters: {count_trainable_parameters(model)/1000000:.2f}M"
    )

    aug_base_path = args.aug_base_path if hasattr(args, "aug_base_path") else None

    logger.warning(f"CLIP Frame Rate: {model.config.clip_frame_rate}")
    preprocessor = Preprocessor(
        dac_base_path=args.dac_base_path,
        clip_base_path=args.clip_base_path,
        aug_base_path=aug_base_path,
        seconds_to_use=args.seconds_to_use,
        dac_pad_token_id=model.config.dac_pad_token_id,
        dac_num_codebooks=model.config.num_rvq,
        dac_frame_rate=model.config.dac_frame_rate,
        clip_frame_rate=model.config.clip_frame_rate,
        label_pad_token_id=-100,
    )

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].with_transform(
            preprocessor.preprocess_train,
        )

        eval_dataset = raw_datasets_eval["validation"].map(
            preprocessor.preprocess_eval,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Praparing validation dataset",
        )
        eval_dataset.set_format(
            "pt",
            columns=[
                "inputs_embeds",
                "decoder_input_ids",
                "labels",
            ],
        )
        demo_dataset = raw_datasets_eval["demo"].map(
            preprocessor.preprocess_eval,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Praparing demo dataset",
        )
        demo_dataset.set_format("pt", columns=["file_path", "inputs_embeds"])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
    )
    demo_dataloader = DataLoader(demo_dataset, batch_size=1)

    # Load DAC to generate audio samples
    if accelerator.is_main_process:
        if config.dac_sample_rate == 44100:
            dac_model_path = dac.utils.download(model_type="44khz")
        elif config.dac_sample_rate == 16000:
            dac_model_path = dac.utils.download(model_type="16khz")
        dac_model = dac.DAC.load(dac_model_path).cuda()
        dac_model.eval()
    else:
        demo_metadata = None

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler_type == "inverse_sqrt" and hasattr(args, "time_scale"):
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            timescale=args.time_scale,
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # Prepare everything with our `accelerator`.
    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler) = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and isinstance(checkpointing_steps, int):
        checkpointing_steps = int(checkpointing_steps)

    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        now = datetime.now()
        formatted_datetime = now.strftime("%Y%m%d-%H%M")
        args.loggigng_dir = os.path.join(args.logging_dir, formatted_datetime)
        accelerator.init_trackers(args.logging_dir)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    if args.split_batches:
        total_batch_size = int(total_batch_size / accelerator.num_processes)

    logger.warning("***** Running training *****")
    logger.warning(f"  Num examples = {len(train_dataset)}")
    logger.warning(f"  Num Epochs = {args.num_train_epochs}")
    logger.warning(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.warning(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.warning(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    logger.warning(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if not args.overwrite_output_dir and os.path.exists(
        os.path.join(args.output_dir, "checkpoints")
    ):
        if args.resume_from_checkpoint is not None:
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [
                f
                for f in os.scandir(os.path.join(args.output_dir, "checkpoints"))
                if f.is_dir()
            ]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ].name  # Sorts folders by date modified, most recent checkpoint is the last
            accelerator.print(f"Resumed from checkpoint: {dirs[-1]}")
            accelerator.load_state(dirs[-1])
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    resume_init_step = completed_steps

    # update the progress_bar if load from checkpoint
    if args.with_tracking:
        total_loss = 0.0
        logging_loss = 0.0
        before_epoch_loss = 0.0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        logger.warning(f"***** Running epoch {epoch} *****")
        epoch_iterator = tqdm(
            active_dataloader,
            desc="Training",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            colour="CYAN",
        )
        for step, batch in enumerate(epoch_iterator):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    completed_steps += 1
                    # Add loss information to tqdm
                    epoch_iterator.set_postfix(
                        loss=total_loss / (completed_steps - resume_init_step)
                    )

                    if completed_steps % args.logging_steps == 0:
                        train_log = {
                            "train/learning_rate": lr_scheduler.get_last_lr()[0]
                        }
                        train_log["train/0_loss_total"] = (
                            total_loss - logging_loss
                        ) / args.logging_steps
                        logging_loss = total_loss

                        # Log loss for each codebooks
                        for idx, codebook_loss in enumerate(outputs.codebook_loss):
                            train_log[f"train/loss_codebook_{idx}"] = codebook_loss

                        accelerator.log(train_log, step=completed_steps)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(
                            args.output_dir, "checkpoints", output_dir
                        )
                    accelerator.save_state(output_dir, safe_serialization=False)
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.config.save_pretrained(output_dir)
                    validation_loss(
                        model,
                        eval_dataloader,
                        accelerator,
                        args,
                        config,
                        logging_step=completed_steps,
                    )
                    if accelerator.is_main_process:
                        generate_audio(
                            model,
                            dac_model,
                            demo_dataloader,
                            demo_metadata,
                            accelerator,
                            config,
                            logging_step=completed_steps,
                        )
                    # accelerator.wait_for_everyone()

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, "checkpoints", output_dir)
            accelerator.save_state(output_dir, safe_serialization=False)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.config.save_pretrained(output_dir)
            validation_loss(
                model,
                eval_dataloader,
                accelerator,
                args,
                config,
                logging_step=epoch,
            )
            if accelerator.is_main_process:
                generate_audio(
                    model,
                    dac_model,
                    demo_dataloader,
                    demo_metadata,
                    accelerator,
                    config,
                    logging_step=epoch,
                )

        if args.with_tracking:
            result = {}
            result["train/0_loss_epoch_train"] = (total_loss - before_epoch_loss) / len(
                train_dataloader
            )
            result["train/steps"] = completed_steps

            before_epoch_loss = total_loss
            accelerator.log(result, step=epoch)


def validation_loss(
    model, eval_dataloader, accelerator, args, config, logging_step=None
):
    model.eval()
    eval_iterator = tqdm(
        eval_dataloader,
        desc="Validation",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        colour="MAGENTA",
    )
    validation_loss = 0
    codebook_loss_list = [0 for i in range(config.num_rvq)]
    for step, batch in enumerate(eval_iterator):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            validation_loss += loss.item()
            for idx, codebook_loss in enumerate(outputs.codebook_loss):
                codebook_loss_list[idx] += codebook_loss

    validation_loss = validation_loss / len(eval_dataloader)
    logger.warning(f"Validation loss: {validation_loss:.4f}")

    if args.with_tracking:
        result = {}
        result["valid/0_loss_total"] = validation_loss
        for i in range(config.num_rvq):
            result[f"valid/loss_codebook_{i}"] = codebook_loss_list[i] / len(
                eval_dataloader
            )
        accelerator.log(result, step=logging_step)
    model.train()


def generate_audio(
    model,
    dac_model,
    demo_dataloader,
    demo_metadata,
    accelerator,
    config,
    logging_step=None,
):
    model.eval()
    # Log generated audio
    demo_iterator = tqdm(
        demo_dataloader,
        desc="Generating Audio Samples",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        colour="GREEN",
    )
    tb_tracker = accelerator.get_tracker("tensorboard", unwrap=True)
    with torch.no_grad():
        for step, batch in enumerate(demo_iterator):
            with torch.autocast("cuda", torch.float16):
                if config.classifier_free_guidance:
                    generated_codes = accelerator.unwrap_model(model).generate_cfg(
                        batch["inputs_embeds"].cuda(), do_sample=True, top_k=256
                    )
                else:
                    generated_codes = accelerator.unwrap_model(model).generate(
                        batch["inputs_embeds"].cuda(), do_sample=True, top_k=256
                    )
                channel = generated_codes[:, :, : config.num_rvq]
                channel = channel.transpose(-1, -2)
                z = dac_model.quantizer.from_codes(channel)[0]
                audio = dac_model.decode(z).squeeze(0).squeeze(0).to(torch.float32)
                if demo_metadata:
                    vid = demo_metadata[batch["file_path"][0].strip()]
                else:
                    vid = batch["file_path"][0].replace(".npy", "")

                tb_tracker.add_audio(
                    tag=f"samples/{vid}",
                    snd_tensor=audio,
                    global_step=logging_step,
                    sample_rate=config.dac_sample_rate,
                )
    model.train()


if __name__ == "__main__":
    main()
