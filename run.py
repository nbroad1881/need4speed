import time
from datetime import datetime
from pathlib import Path
from typing import List, Union
import warnings
import logging
import yaml
from functools import partial

import fire
import wandb
import numpy as np
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainerCallback,
    default_data_collator,
    DataCollatorWithPadding,
)


warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

DEFAULT = {
    "per_device_train_batch_size": 8,
    "tf32": False,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": True,
    "group_by_length": False,
    "gradient_checkpointing": False,
    "torch_compile": False,
    "optim": "adamw_torch",
    "mixed_precision": "fp32",
    "fp16": False,
    "bf16": False,
    "model_name_or_path": "roberta-base",
    "max_seq_length": 256,
    "varied_lengths": False,
    "pad_to_multiple_of": None,
    "padding": "longest",
    "resize_embeddings": 0,
}

NUM_SAMPLES = 10_000


def create_dataset(num_samples, max_seq_length, varied_lengths=False):
    """Create a dummy dataset for testing purposes."""

    input_ids = []

    for _ in range(num_samples):
        if varied_lengths:
            length = np.random.randint(1, max_seq_length)
        else:
            length = max_seq_length

        input_ids.append(np.random.randint(200, 300, length))

    attention_mask = [[1] * len(input_id) for input_id in input_ids]

    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [0] * len(input_ids),
        }
    )


def model_init(resize_embeddings, model_name_or_path):
    """
    Initialize the model.
    Resize the embeddings if necessary.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=2
    )

    if resize_embeddings > 0:
        num_embeds = model.get_input_embeddings().weight.data.size(0)
        if num_embeds % resize_embeddings != 0:
            model.resize_token_embeddings(
                num_embeds + (resize_embeddings - num_embeds % resize_embeddings)
            )

    return model


class Need4SpeedCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        self.counter = 0
        self.start_timer = None
        self.batches_completed = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.counter += 1

        if (
            self.start_timer is None
            and (time.time() - self.epoch_start > 10)
            and (self.counter > 10)
        ):
            self.start_timer = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_timer is not None:
            self.batches_completed += 1

        if (self.start_timer is not None) and (time.time() - self.start_timer) > 10:
            samples_completed = (
                self.batches_completed
                * args.per_device_train_batch_size
                * args.n_gpu
                * args.gradient_accumulation_steps
            )
            time_elapsed = time.time() - self.start_timer
            wandb.log(
                {
                    "samples_per_second": samples_completed / time_elapsed,
                    "time_elapsed": time_elapsed,
                }
            )
            control.should_training_stop = True


def wandb_train_fn():

    with wandb.init() as run:

        config = wandb.config

        params = [
            "per_device_train_batch_size",
            "tf32",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "group_by_length",
            "gradient_checkpointing",
            "torch_compile",
            "optim",
        ]
        sweep_parameters = {
            param: config.get(param, DEFAULT[param]) for param in params
        }

        ds = create_dataset(
            num_samples=NUM_SAMPLES,
            max_seq_length=config.get("max_seq_length", DEFAULT["max_seq_length"]),
            varied_lengths=config.get("varied_lengths", DEFAULT["varied_lengths"]),
        )

        sweep_parameters["fp16"] = False
        if config.get("mixed_precision", DEFAULT["mixed_precision"]) == "fp16":
            sweep_parameters["fp16"] = True
        elif config.get("mixed_precision", DEFAULT["mixed_precision"]) == "bf16":
            sweep_parameters["bf16"] = True

        tokenizer = AutoTokenizer.from_pretrained(
            config.get("model_name_or_path", DEFAULT["model_name_or_path"])
        )

        if config.get("varied_lengths", False):

            data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                pad_to_multiple_of=config.get("pad_to_multiple_of", DEFAULT["pad_to_multiple_of"]),
                max_length=config.get("max_seq_length", DEFAULT["max_seq_length"]),
                padding=config.get("padding", DEFAULT["padding"]),
            )

        else:
            data_collator = default_data_collator

        training_args = TrainingArguments(
            f"sweeps/wandb-sweep-{wandb.run.id}",
            **sweep_parameters,
            save_strategy="no",
            evaluation_strategy="no",
            logging_strategy="no",
            disable_tqdm=False,
            log_level="warning",
            report_to="wandb",
        )

        _model_init = partial(
                model_init,
                resize_embeddings=config.get("resize_embeddings", DEFAULT["resize_embeddings"]),
                model_name_or_path=config.get("model_name_or_path", DEFAULT["model_name_or_path"]),
            )

        trainer = Trainer(
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=ds,
            model_init=_model_init,
            data_collator=data_collator,
            callbacks=[Need4SpeedCallback()],
        )

        trainer.train()


def main(config_path: Union[str, List[str]], n: int = 1):
    """
    Run a sweep with the given config file.
    Repeat `n` times.
    """

    sweep_start_time = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    if config_path == "all":
        config_path = Path("configs").glob("*.yaml")

    elif isinstance(config_path, str):

        config_path = [config_path]

    for config_path in config_path:
        
        with open(config_path) as f:
            sweep_config = yaml.safe_load(f)

        set_seed(42)

        sweep_config["sweep_start_time"] = sweep_start_time

        for _ in range(n):
            sweep_id = wandb.sweep(sweep_config, project="need4speed")
            wandb.agent(sweep_id, wandb_train_fn)


if __name__ == "__main__":
    fire.Fire(main)
