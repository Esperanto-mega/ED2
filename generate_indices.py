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
from index.datasets import EmbDataset

def if_collided(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description = "Index")
    parser.add_argument("--ckpt_path", type = str, default = "", help = "")
    parser.add_argument("--item_data_path", type = str, default = "", help = "")
    parser.add_argument("--user_data_path", type = str, default = "", help = "")
    parser.add_argument("--save_path", type = str, default = "", help = "")
    parser.add_argument("--device_map", type = str, default = "1", help = "gpu or cpu")
    return parser.parse_args()

args = parse_args()
print(args)

device_map = {'': int(args.device_map)}
MODEL = LlamaWithRQ.from_pretrained(args.ckpt_path, torch_dtype = torch.float16, low_cpu_mem_usage = True, device_map = device_map)
MODEL.eval()
device = MODEL.device
postfix = '<p-{}>'

data = EmbDataset(args.item_data_path)
data_loader = DataLoader(data, num_workers = 4, batch_size = 64, shuffle = False, pin_memory = True)
rqvae = MODEL.item_rqvae
prefix = MODEL.prefix

index_table = {}
all_indices = []
all_indices_str = []
with torch.no_grad():
    for x in tqdm(data_loader):
        indices = rqvae.get_indices(x.to(device), False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            if str(code) in index_table:
                index_table[str(code)] += 1
            else:
                index_table[str(code)] = 0
            code.append(postfix.format(index_table[str(code)]))

            all_indices.append(code)
            all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

print("All indices number: ", len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))
print('Re-index number:', max(index_table.values()))

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

reindex_dict = {'reindex': max(index_table.values())}

json_path = os.path.join(args.save_path,'indices.item.json')
with open(json_path, 'w',encoding = 'utf-8') as f:
    json.dump(all_indices_dict, f)

reindex_path = os.path.join(args.save_path,'reindex.item.json')
with open(reindex_path, 'w',encoding = 'utf-8') as f:
    json.dump(reindex_dict, f)

data = EmbDataset(args.user_data_path)
data_loader = DataLoader(data, num_workers = 4, batch_size = 64, shuffle = False, pin_memory = True)
rqvae = MODEL.user_rqvae
prefix = MODEL.user_prefix

# index_table = {}
all_indices = []
all_indices_str = []
with torch.no_grad():
    for x in tqdm(data_loader):
        indices = rqvae.get_indices(x.to(device), False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            # if str(code) in index_table:
            #     index_table[str(code)] += 1
            # else:
            #     index_table[str(code)] = 0
            # code.append(postfix.format(index_table[str(code)]))

            all_indices.append(code)
            all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

print("All indices number: ", len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))
# print('Re-index number:', max(index_table.values()))

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

# reindex_dict = {'reindex': max(index_table.values())}

json_path = os.path.join(args.save_path,'indices.user.json')
with open(json_path, 'w',encoding = 'utf-8') as f:
    json.dump(all_indices_dict, f)

# reindex_path = os.path.join(args.save_path,'reindex.user.json')
# with open(reindex_path, 'w',encoding = 'utf-8') as f:
#     json.dump(reindex_dict, f)