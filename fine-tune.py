import argparse
import os
import sys
from typing import List

import torch
import transformers
from peft import PeftModel
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import *
from collator import Collator

import argparse
from utils import *
from rq_llama import *

parser = argparse.ArgumentParser(description = 'rqllama-finetune')
parser = parse_finetune_args(parser)
args = parser.parse_args()

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

train_data, valid_data = load_finetune_datasets(args)

rqllama = LlamaWithRQ.from_pretrained(args.ckpt_path, torch_dtype = torch.float16, low_cpu_mem_usage = True, device_map = device_map)
tokenizer = rqllama.tokenizer
# PeftModelForCausalLM
model = rqllama.model
device = rqllama.device

postfix = '<p-{}>'
new_tokens = []
new_ids = list(range(args.reindex))
for i in new_ids:
    new_tokens.append(postfix.format(int(i)))
tokenizer.add_tokens(new_tokens)

if local_rank == 0:
    print("token num:", len(rqllama.tokenizer))
    print("data num:", len(train_data))

collator = Collator(args, tokenizer)

# Re-index Embedding
new_ids = torch.tensor(new_ids, dtype = torch.float16).reshape(-1,1)
re_index_emb = torch.nn.Linear(1, model.config.hidden_size, dtype = torch.float16).to(device)
new_embeddings = re_index_emb(new_ids.to(device))
# PeftModelForCausalLM -> LlamaForCausalLM -> LlamaModel
model.model.model.embed_tokens.original_module.weight.data = torch.cat([model.model.model.embed_tokens.original_module.weight.data, new_embeddings], dim = 0)
model.model.model.embed_tokens.modules_to_save.default.weight.data = torch.cat([model.model.model.embed_tokens.modules_to_save.default.weight.data, new_embeddings], dim = 0)

new_lm_head = torch.randn(args.reindex, model.config.hidden_size, requires_grad = True).to(device)
# print('new_lm_head:',new_lm_head.requires_grad)
# PeftModelForCausalLM -> LlamaForCausalLM
model.model.lm_head.original_module.weight.data = torch.cat([model.model.lm_head.original_module.weight.data, new_lm_head], dim = 0)
model.model.lm_head.modules_to_save.default.weight.data = torch.cat([model.model.lm_head.modules_to_save.default.weight.data, new_lm_head], dim = 0)

model.config.vocab_size = len(tokenizer)

# print(model.model.model.embed_tokens.original_module.weight.shape)
# print(len(tokenizer))

model.train()

if local_rank == 0:
    model.print_trainable_parameters()

trainer = transformers.Trainer(
    model = model,
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
        save_total_limit = 50,
        load_best_model_at_end = True,
        deepspeed = args.deepspeed,
        ddp_find_unused_parameters = False if ddp else None,
        report_to = None,
        eval_delay = 1 if args.save_and_eval_strategy=="epoch" else 2000,
        dataloader_num_workers = args.dataloader_num_workers,
        dataloader_prefetch_factor = args.dataloader_prefetch_factor,
        remove_unused_columns = args.remove_unused_columns,
    ),
    tokenizer = tokenizer,
    data_collator = collator,
)
model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)

trainer.save_state()
trainer.save_model(output_dir = args.output_dir)
