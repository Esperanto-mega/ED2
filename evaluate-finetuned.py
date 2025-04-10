import argparse
import json
import os
import sys

import torch
import transformers
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_topk_results, get_metrics_results

parser = argparse.ArgumentParser(description = 'rqllama-evaluate')
parser = parse_evaluate_args(parser)
args = parser.parse_args()

set_seed(args.seed)
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK") or 0)
torch.cuda.set_device(local_rank)
if local_rank == 0:
    print(vars(args))

dist.init_process_group(backend = "nccl", world_size = world_size, rank = local_rank)

device_map = {"": local_rank}
device = torch.device("cuda",local_rank)

tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
base_model = LlamaForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage = True, device_map = device_map)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype = torch.float16, device_map = device_map)

model = DistributedDataParallel(model, device_ids = [local_rank])

if args.test_prompt_ids == "all":
    if args.test_task.lower() == "seqrec":
        prompt_ids = range(len(all_prompt["seqrec"]))
    elif args.test_task.lower() == "itemsearch":
        prompt_ids = range(len(all_prompt["itemsearch"]))
    elif args.test_task.lower() == "fusionseqrec":
        prompt_ids = range(len(all_prompt["fusionseqrec"]))
else:
    prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

test_data = load_test_dataset(args)
if local_rank == 0:
    print("evaluate data num:", len(test_data))
ddp_sampler = DistributedSampler(test_data, num_replicas = world_size, rank = local_rank, drop_last = True)
collator = TestCollator(args, tokenizer)
all_items = test_data.get_all_items()
prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)
test_loader = DataLoader(
    test_data, 
    batch_size = args.test_batch_size, 
    collate_fn = collator,
    sampler = ddp_sampler, 
    num_workers = 4, 
    pin_memory = True
)

model.eval()

metrics = args.metrics.split(",")
all_prompt_results = []

print('prompts:', len(prompt_ids))

with torch.no_grad():
    for prompt_id in prompt_ids:
        if local_rank == 0:
            print("Start prompt: ",prompt_id)
        test_loader.dataset.set_prompt(prompt_id)
        metrics_results = {}
        total = 0

        for step, batch in enumerate(tqdm(test_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]
            bs = len(targets)
            num_beams = args.num_beams

            while True:
                try:
                    output = model.module.generate(
                            input_ids = inputs["input_ids"],
                            attention_mask = inputs["attention_mask"],
                            max_new_tokens = 10,
                            prefix_allowed_tokens_fn = prefix_allowed_tokens,
                            num_beams = num_beams,
                            num_return_sequences = num_beams,
                            output_scores = True,
                            return_dict_in_generate = True,
                            early_stopping = True,
                        )
                    break
                except torch.cuda.OutOfMemoryError as e:
                    print("Out of memory!")
                    num_beams = num_beams -1
                    print("Beam:", num_beams)
                except Exception:
                    raise RuntimeError
            output_ids = output["sequences"]
            scores = output["sequences_scores"]

            # output_ids.shape: torch.Size([20, 101])
            # scores.shape: torch.Size([20])

            output = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
            # output.length: 20
            '''
            Below is an instruction that describes a task. 
            Write a response that appropriately completes the request.\n\n
            ### Instruction:\nThe user has interacted with items  <a-213> <b-171> <c-26> <d-74> <p-0> ,  <a-14> <b-33> <c-196> <d-121> <p-0> ,  
            <a-213> <b-23> <c-128> <d-13> <p-8> ,  <a-1> <b-23> <c-68> <d-71> <p-1> in chronological order. 
            Can you predict the next possible item that the user may expect?\n\n
            ### Response: <a-9> <b-23> <c-123> <d-85> <p-2>
            '''

            topk_res = get_topk_results(
                output, 
                scores, 
                targets, 
                num_beams,
                all_items = all_items if args.filter_items else None
            )

            bs_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=bs, object_list=bs_gather_list)
            total += sum(bs_gather_list)
            res_gather_list = [None for _ in range(world_size)]
            dist.all_gather_object(obj=topk_res, object_list=res_gather_list)

            if local_rank == 0:
                all_device_topk_res = []
                for ga_res in res_gather_list:
                    all_device_topk_res += ga_res
                batch_metrics_res = get_metrics_results(all_device_topk_res, metrics)
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res
                
                if (step + 1) % 50 == 0:
                    temp = {}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)
            dist.barrier()

        if local_rank == 0:
            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")
        dist.barrier()
dist.barrier()

if local_rank == 0:
    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")

    save_data={}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent = 4)
    print("Save file: ", args.results_file)
