import os
import collections
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from rq_llama import *

def parse_args():
    parser = argparse.ArgumentParser(description = "Index")
    parser.add_argument("--ckpt_path", type = str, default = "", help = "")
    parser.add_argument("--item_save_path", type = str, default = "", help = "")
    parser.add_argument("--user_save_path", type = str, default = "", help = "")
    parser.add_argument("--device_map", type = str, default = "1", help = "gpu or cpu")
    return parser.parse_args()

args = parse_args()
print(args)
device_map = {'': int(args.device_map)}
MODEL = LlamaWithRQ.from_pretrained(args.ckpt_path, torch_dtype = torch.float16, low_cpu_mem_usage = True, device_map = device_map)
MODEL.eval()
device = MODEL.device
llama = MODEL.model.get_decoder()
tokenizer = MODEL.tokenizer
item_texts = MODEL.item_texts
user_texts = MODEL.user_texts

all_idx = []
all_embeddings = []
with torch.no_grad():
    for idx, text in tqdm(item_texts.items()):
        item_text = text['title'] + ' ' + text['description']
        item_ids = tokenizer(item_text, return_tensors = 'pt', padding = True, truncation = True).to(device)
        item_emb = llama(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
        item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
        item_emb = item_emb.sum(dim = 1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)

        all_idx.append(idx)
        all_embeddings.append(item_emb.detach().cpu().numpy().flatten().tolist())
    
results = {
    'id': all_idx,
    'emb': []
}

for emb in tqdm(all_embeddings):
    str_emb = ''
    for e in emb:
        str_emb = str_emb + str(e) + ' '
    results['emb'].append(str_emb[:-1])

df = pd.DataFrame(results)
df.to_csv(args.item_save_path, sep = '\t', header = 0, index = False)

all_idx = []
all_embeddings = []
with torch.no_grad():
    for idx, text in tqdm(user_texts.items()):
        user_text = ' '.join(text)
        user_ids = tokenizer(user_text, return_tensors = 'pt', padding = True, truncation = True).to(device)
        user_emb = llama(input_ids = user_ids.input_ids, attention_mask = user_ids.attention_mask)
        user_emb = user_emb.last_hidden_state * user_ids.attention_mask.unsqueeze(-1)
        user_emb = user_emb.sum(dim = 1) / user_ids.attention_mask.sum(dim = -1, keepdim = True)

        all_idx.append(idx)
        all_embeddings.append(user_emb.detach().cpu().numpy().flatten().tolist())

results = {
    'id': all_idx,
    'emb': []
}

for emb in tqdm(all_embeddings):
    str_emb = ''
    for e in emb:
        str_emb = str_emb + str(e) + ' '
    results['emb'].append(str_emb[:-1])

df = pd.DataFrame(results)
df.to_csv(args.user_save_path, sep = '\t', header = 0, index = False)