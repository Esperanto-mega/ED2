import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration

class VanillaCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
    def __call__(self, data):
        # print('collator data:',data)
        '''
        [{
        'input_ids': 
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
            ### Instruction:\n
                Access the user's historical item interaction records: {inters}. 
                Your objective is to describe the next potential item for him, taking into account his past interactions.\n\n
            ### Response:", 
        'labels': 
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
            ### Instruction:\n
                Access the user's historical item interaction records: {inters}. 
                Your objective is to describe the next potential item for him, taking into account his past interactions.\n\n
            ### Response:
                Dunlop guitar picks are a top choice of today's pro musician! Dunlop's wide variety of gauges, shapes, sizes and materials 
                allows the player to select the exact pick for his/her own particular style of playing. From classic country to nu-metal, 
                every great player knows that their pick is an integral part of their tone, and Dunlop guitar picks are the picks that more 
                pros rely on in the studio or on stage. Picks are a grossly underrated accessory. Don't sacrifice your tone...pick Dunlop guitar picks!.",
        'inters': '341,2804,3895,3893,7064', 
        'item': 'placeholder', 
        'task': 'inters2description'
        }, 
        {
        'input_ids': 
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
            ### Instruction:\n
                Based on the user\'s historical interactions with the following items: {inters}. 
                You can infer his preference by observing the historical interactions: "The user\'s short-term preferences have shift to heavier picks, 
                suggesting that He is looking for a heavier sound.". Now the user wants a new item and searches for: "I like the durability and 
                effectiveness of the picks.". Please select a suitable item that matches his preference and search intent.\n\n
            ### Response:', 
        'labels': 
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
            ### Instruction:\n
                Based on the user\'s historical interactions with the following items: {inters}. 
                You can infer his preference by observing the historical interactions: "The user\'s short-term preferences have shift to heavier picks, 
                suggesting that He is looking for a heavier sound.". Now the user wants a new item and searches for: "I like the durability and 
                effectiveness of the picks.". Please select a suitable item that matches his preference and search intent.\n\n
            ### Response:{item}', 
        'inters': '122,469,8918', 
        'item': '7140', 
        'task': 'itemsearch'
        }]
        '''
        dict_data = {
            'input_ids': [],
            'labels': [],
            'inters': [],
            'item': [],
            'users': [],
            'user': [],
            'task': []
        }

        for d in data:
            for k in dict_data.keys():
                if k == 'labels':
                    dict_data[k].append(d[k] + self.tokenizer.eos_token)
                else:
                    dict_data[k].append(d[k])

        return dict_data

class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        full_texts = [d["labels"] + self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(
            text = full_texts,
            text_target = input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs["input_ids"])
        if self.only_train_response:
            # ignore padding
            labels[labels == self.tokenizer.pad_token_id] = -100
            # ignore input text
            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100

        inputs["labels"] = labels

        return inputs

class TestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        if isinstance(self.tokenizer, LlamaTokenizer):
            self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]
        inputs = self.tokenizer(
            text = input_texts,
            return_tensors ="pt",
            padding = "longest",
            max_length = self.tokenizer.model_max_length,
            truncation = True,
            return_attention_mask = True,
        )

        return (inputs, targets)

# RuntimeError: Cannot re-initialize CUDA in forked subprocess. 
# To use CUDA with multiprocessing, you must use the 'spawn' start method.
# class ValidCollator(object):
#     def __init__(self, args, model):
#         self.args = args
#         self.model = model
#         self.only_train_response = args.only_train_response
#         self.tokenizer = model.tokenizer
#     def __call__(self, data):
#         llama_model = self.model.model.get_decoder()
#         for d in data:
#             inter_emb_list = []
#             inter_item_list = d['inters'].split(',')
#             for inter_item in inter_item_list:
#                 inter_feature = self.model.item_texts[inter_item]['title'] + ' ' + self.model.item_texts[inter_item]['description']
#                 inter_id = self.tokenizer(inter_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
#                 inter_emb = llama_model(input_ids = inter_id.input_ids, attention_mask = inter_id.attention_mask)
#                 inter_emb = inter_emb.last_hidden_state * inter_id.attention_mask.unsqueeze(-1)
#                 inter_emb = inter_emb.sum(dim=1) / inter_id.attention_mask.sum(dim = -1, keepdim = True)
#                 inter_emb_list.append(inter_emb.detach())
#             inter_embs = torch.cat(inter_emb_list, dim = 0)
#             item_feature = self.model.item_texts[d['item']]['title'] + ' ' + self.model.item_texts[d['item']]['description']
#             item_ids = self.tokenizer(item_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
#             item_emb = llama_model(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
#             item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
#             item_emb = item_emb.sum(dim=1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)
#             item_emb = item_emb.detach()

#             rqids = self.model.rqvae.get_indices(torch.cat([inter_embs, item_emb], dim = 0))

#             inters_rqids = rqids.view(-1, rqids.shape[-1]).cpu().numpy().tolist()[:-1]
#             item_rqid = rqids.view(-1, rqids.shape[-1]).cpu().numpy().tolist()[-1]

#             text_rqids = {}
#             code = ''
#             for rqid in inters_rqids:
#                 for k, idx in enumerate(rqid):
#                     code = code + self.model.prefix[k].format(idx)
#                 code = code + ', '
#             text_rqids['inters'] = code[:-2]
#             code = ''
#             for k, idx in enumerate(item_rqid):
#                 code = code + self.model.prefix[k].format(idx)
#             text_rqids['item'] = code

#             d['input_ids'] = d['input_ids'].format(inters = text_rqids['inters'])
#             d['labels'] = d['labels'].format(inters = text_rqids['inters'], item = text_rqids['item'])

#         input_texts = [d["input_ids"] for d in data]
#         full_texts = [d["labels"] + self.tokenizer.eos_token for d in data]

#         inputs = self.tokenizer(
#             text = full_texts,
#             text_target = input_texts,
#             return_tensors="pt",
#             padding="longest",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_attention_mask=True,
#         )

#         labels = copy.deepcopy(inputs["input_ids"])
#         if self.only_train_response:
#             labels[labels == self.tokenizer.pad_token_id] = -100
#             labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100
#         inputs["labels"] = labels

#         return inputs

# RuntimeError: Cannot re-initialize CUDA in forked subprocess. 
# To use CUDA with multiprocessing, you must use the 'spawn' start method.
# class TestCollator(object):
#     def __init__(self, args, model):
#         self.args = args
#         self.model = model
#         self.tokenizer = model.tokenizer
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = 0
#         if isinstance(self.tokenizer, LlamaTokenizer):
#             self.tokenizer.padding_side = "left"
    
#     def __call__(self, data):
#         llama_model = self.model.model.get_decoder()
#         for d in data:
#             inter_emb_list = []
#             inter_item_list = d['inters'].split(',')
#             for inter_item in inter_item_list:
#                 inter_feature = self.model.item_texts[inter_item]['title'] + ' ' + self.model.item_texts[inter_item]['description']
#                 inter_id = self.tokenizer(inter_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
#                 inter_emb = llama_model(input_ids = inter_id.input_ids, attention_mask = inter_id.attention_mask)
#                 inter_emb = inter_emb.last_hidden_state * inter_id.attention_mask.unsqueeze(-1)
#                 inter_emb = inter_emb.sum(dim=1) / inter_id.attention_mask.sum(dim = -1, keepdim = True)
#                 inter_emb_list.append(inter_emb.detach())
#             inter_embs = torch.cat(inter_emb_list, dim = 0)
#             item_feature = self.model.item_texts[d['item']]['title'] + ' ' + self.model.item_texts[d['item']]['description']
#             item_ids = self.tokenizer(item_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
#             item_emb = llama_model(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
#             item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
#             item_emb = item_emb.sum(dim=1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)
#             item_emb = item_emb.detach()

#             rqids = self.model.rqvae.get_indices(torch.cat([inter_embs, item_emb], dim = 0))

#             inters_rqids = rqids.view(-1, rqids.shape[-1]).cpu().numpy().tolist()[:-1]
#             item_rqid = rqids.view(-1, rqids.shape[-1]).cpu().numpy().tolist()[-1]

#             text_rqids = {}
#             code = ''
#             for rqid in inters_rqids:
#                 for k, idx in enumerate(rqid):
#                     code = code + self.model.prefix[k].format(idx)
#                 code = code + ', '
#             text_rqids['inters'] = code[:-2]
#             code = ''
#             for k, idx in enumerate(item_rqid):
#                 code = code + self.model.prefix[k].format(idx)
#             text_rqids['item'] = code

#             d['input_ids'] = d['input_ids'].format(inters = text_rqids['inters'])
#             d['labels'] = d['labels'].format(inters = text_rqids['inters'], item = text_rqids['item'])
        
#         input_texts = [d["input_ids"] for d in data]
#         targets = [d["labels"] for d in data]

#         inputs = self.tokenizer(
#             text=input_texts,
#             return_tensors="pt",
#             padding="longest",
#             max_length=self.tokenizer.model_max_length,
#             truncation=True,
#             return_attention_mask=True,
#         )

#         return (inputs, targets)