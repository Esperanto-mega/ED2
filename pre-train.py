import os
import sys
from typing import List
import argparse

import wandb
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from collator import VanillaCollator
from rq_llama import *
from utils import *

parser = argparse.ArgumentParser(description = 'rqllama-pretrain')
parser = parse_global_args(parser)
parser = parse_train_args(parser)
parser = parse_dataset_args(parser)
parser = parse_rqvae_args(parser)
args = parser.parse_args()
wandb.init(config = args, reinit = True)

set_seed(args.seed)
ensure_dir(args.output_dir)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
local_rank = int(os.environ.get("LOCAL_RANK") or 0)
if local_rank == 0:
    print(vars(args))
if ddp:
    device_map = {"": local_rank}

train_data, valid_data = load_datasets(args)

config = LlamaConfig.from_pretrained(args.base_model)
config.args = vars(args)
rqllama = LlamaWithRQ(config)

ckpt = torch.load(args.item_model, map_location = torch.device('cpu'))
state_dict = ckpt["state_dict"]
rqllama.item_rqvae.load_state_dict(state_dict)
for i in range(len(args.num_emb_list)):
    rqllama.item_rqvae.rq.vq_layers[i].initted = True
ckpt = torch.load(args.user_model, map_location = torch.device('cpu'))
state_dict = ckpt["state_dict"]
rqllama.user_rqvae.load_state_dict(state_dict)
for i in range(len(args.num_emb_list)):
    rqllama.user_rqvae.rq.vq_layers[i].initted = True

if local_rank == 0:
    print("token num:", len(rqllama.tokenizer))
    print("data num:", len(train_data))
    rqllama.tokenizer.save_pretrained(args.output_dir)
    rqllama.config.save_pretrained(args.output_dir)

if args.resume_from_checkpoint:
    checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
    args.resume_from_checkpoint = False
    if os.path.exists(checkpoint_name):
        if local_rank == 0:
            print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        rqllama.model = set_peft_model_state_dict(rqllama.model, adapters_weights)
    else:
        if local_rank == 0:
            print(f"Checkpoint {checkpoint_name} not found")

if local_rank == 0:
    rqllama.model.print_trainable_parameters()

if not ddp and torch.cuda.device_count() > 1:
    rqllama.is_parallelizable = True
    rqllama.model_parallel = True

collator = VanillaCollator(args, rqllama.tokenizer)

trainer = transformers.Trainer(
    model = rqllama,
    train_dataset = train_data,
    eval_dataset = valid_data,
    args = transformers.TrainingArguments(
        seed = args.seed,
        per_device_train_batch_size = args.per_device_batch_size,
        per_device_eval_batch_size = args.per_device_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_ratio = args.warmup_ratio,
        num_train_epochs = args.epochs,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        fp16 = args.fp16,
        bf16 = args.bf16,
        logging_steps = args.logging_step,
        optim = args.optim,
        gradient_checkpointing = True,
        evaluation_strategy = args.save_and_eval_strategy,
        save_strategy = args.save_and_eval_strategy,
        eval_steps = args.save_and_eval_steps,
        save_steps = args.save_and_eval_steps,
        output_dir = args.output_dir,
        save_total_limit = 5,
        load_best_model_at_end = True,
        deepspeed = args.deepspeed,
        ddp_find_unused_parameters = False if ddp else None,
        report_to = None,
        eval_delay = 1 if args.save_and_eval_strategy=="epoch" else 2000,
        dataloader_num_workers = args.dataloader_num_workers,
        dataloader_prefetch_factor = args.dataloader_prefetch_factor,
        remove_unused_columns = args.remove_unused_columns,
    ),
    tokenizer = rqllama.tokenizer,
    data_collator = collator,
)
rqllama.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    rqllama = torch.compile(rqllama)

trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)

trainer.save_state()
trainer.save_model(output_dir = args.output_dir)
