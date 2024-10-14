import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
from prompt import sft_prompt, all_prompt
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.user_index_file = args.user_index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = tokenizer("Response:")["input_ids"][1:]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):
        raise NotImplementedError

class UserFeatDataset(BaseDataset):
    def __init__(self, args, task = "pref2user", prompt_sample_num = 1, sample_num = -1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        self._load_data()
        self.feat_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            user_feat = json.load(f)
            # >>> user_feat.keys()
            # dict_keys(['user_explicit_preference', 'user_vague_intention'])
            self.user_feat = user_feat['user_explicit_preference']
            # >>> user_feat['0']
            # ['The user is a passionate musician who enjoys exploring different types of musical instruments.']
            # >>> len(user_feat)
            # 24772

    def _process_data(self):
        feat_data = []
        for uid in self.user_feat:
            one_data = {}
            one_data['user'] = uid

            preference = " ".join(self.user_feat[uid])
            preference = preference.strip().strip(".!?,;:`")
            preference = preference.replace('{','').replace('}','')
            one_data['preference'] = preference

            feat_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace = False)
            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data

    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]
        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        if self.task == 'pref2user':
            instruction = prompt['instruction'].format(preference = d['preference'])
            input = sft_prompt.format(instruction = instruction, response = "")
            output = sft_prompt.format(instruction = instruction, response = prompt["response"])
            return dict(
                input_ids = input, 
                labels = output, 
                inters = 'placeholder', 
                item = 'placeholder',
                users = 'placeholder',
                user = d['user'], 
                task = self.task
            )
        elif self.task == 'user2pref':
            input = sft_prompt.format(instruction = prompt["instruction"], response = "")
            response = prompt["response"].format(preference = d['preference'])
            output = sft_prompt.format(instruction = prompt["instruction"], response = response)
            return dict(
                input_ids = input, 
                labels = output, 
                inters = 'placeholder', 
                item = 'placeholder',
                users = 'placeholder',
                user = d['user'], 
                task = self.task
            )
        else:
            raise NotImplementedError

class UserSearchDataset(BaseDataset):
    def __init__(self, args, prompt_sample_num = 1, prompt_id = 0, sample_num = -1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["usersearch"]

        self._load_data()
        self.search_data = self._process_data()
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.user.json"), 'r') as f:
            self.user_inters = json.load(f)

    def _process_data(self):
        search_data = []
        for iid in self.user_inters.keys():
            users = self.user_inters[iid]
            for i in range(1, len(users)):
                one_data = {}
                one_data['item'] = iid
                one_data['user'] = str(users[i])
                history = users[:i]

                if len(history) > self.max_his_len:
                    history = history[-self.max_his_len:]

                one_data['users'] = ''
                for user in history:
                    one_data['users'] = one_data['users'] + str(user) + ','
                one_data['users'] = one_data['users'][:-1]

                search_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace = False)
            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def __len__(self):
        return len(self.search_data) * self.prompt_sample_num

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.search_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)
        prompt = self.prompts[prompt_id]

        input = sft_prompt.format(instruction = prompt["instruction"], response = "")
        output = sft_prompt.format(instruction = prompt["instruction"], response = prompt["response"])
        return dict(
            input_ids = input,
            labels = output,
            inters = 'placeholder',
            item = d['item'], 
            users = d['users'],
            user = d['user'],
            task = 'usersearch'
        )

# =====================================================================================================================
# seqrec,itemsearch,inters2title,inters2description,preferenceobtain,item2index,index2item,intertitles2item,query2item
# =====================================================================================================================

# seqrec
class SeqRecDataset(BaseDataset):
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["seqrec"]

        self._load_data()
        
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
            # self.inter_data = self.inter_data[:10]
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            # self.inter_data = self.inter_data[:10]
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
            # self.inter_data = self.inter_data[:10]
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data['user'] = uid
                one_data['item'] = str(items[i])
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                one_data['inters'] = ''
                for item in history:
                    one_data['inters'] = one_data['inters'] + str(item) + ','
                one_data['inters'] = one_data['inters'][:-1]

                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            one_data = dict()
            items = self.inters[uid]
            one_data['user'] = uid
            one_data['item'] = str(items[-2])
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data['inters'] = ''
            for item in history:
                one_data['inters'] = one_data['inters'] + str(item) + ','
            one_data['inters'] = one_data['inters'][:-1]
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):
        with open(self.index_file, 'r') as f:
            self.indices = json.load(f)
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

        with open(self.user_index_file, 'r') as f:
            self.user_indices = json.load(f)
        self.remapped_users = dict()
        for uid in self.inters:
            new_user= ''.join(self.user_indices[uid])
            self.remapped_users[uid] = new_user
        
        inter_data = []
        for uid in self.remapped_inters:
            one_data = dict()
            one_data['user'] = self.remapped_users[uid]
            items = self.remapped_inters[uid]
            one_data['item'] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        # for uid in self.inters:
        #     one_data = dict()
        #     items = self.inters[uid]
        #     one_data["item"] = str(items[-1])
        #     history = items[:-1]
        #     if self.max_his_len > 0:
        #         history = history[-self.max_his_len:]
        #     one_data['inters'] = ''
        #     for item in history:
        #         one_data['inters'] = one_data['inters'] + str(item) + ','
        #     one_data['inters'] = one_data['inters'][:-1]
        #     inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace = False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError
                    
    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                for prompt_id in prompt_ids:
                    prompt = self.prompts[prompt_id]
                    input = sft_prompt.format(instruction = prompt["instruction"], response = "")
                    output = sft_prompt.format(instruction = prompt["instruction"], response = prompt["response"])
                    self.valid_text_data.append({
                        "input_ids": input, 
                        "labels": output, 
                        "inters": d['inters'], 
                        "item": d['item'],
                        "users": 'placeholder',
                        "user": d['user'],
                        "task": 'seqrec'})
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                input = sft_prompt.format(instruction = prompt["instruction"], response = "")
                output = sft_prompt.format(instruction = prompt["instruction"], response = prompt["response"])
                self.valid_text_data.append({
                    "input_ids": input, "labels": output, 
                    "inters": d['inters'], "item": d['item'], "users": 'placeholder', "user": d['user'], 
                    "task": 'seqrec'})

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):
        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id
            prompt = self.prompts[prompt_id]
            instruction = prompt["instruction"].format(**d)
            response = prompt["response"].format(**d)
            input = sft_prompt.format(instruction = instruction, response = "")
            return dict(input_ids = input, labels = response)
            # output = prompt["response"]
            # return dict(input_ids = input, labels = output, inters = d['inters'], item = d['item'], task = 'seqrec')

        prompt = self.prompts[prompt_id]

        input = sft_prompt.format(instruction = prompt["instruction"], response = "")
        output = sft_prompt.format(instruction = prompt["instruction"], response = prompt["response"])

        return dict(input_ids = input, labels = output, inters = d['inters'], item = d['item'], user = d['user'],  task = 'seqrec', users = 'placeholder')

# itemsearch & query2item
class ItemSearchDataset(BaseDataset):
    def __init__(self, args, mode="train", task = 'itemsearch',
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.task = task.lower()
        self.prompts = all_prompt[self.task]

        self._load_data()
        self.search_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)

    def _process_data(self):
        search_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]
        user_vague_intention = self.user_info["user_vague_intention"]
        if self.mode == 'train':
            user_vague_intention = user_vague_intention["train"]
        elif self.mode == 'test':
            user_vague_intention = user_vague_intention["test"]
        else:
            raise NotImplementedError

        for uid in user_explicit_preference.keys():
            one_data = {}
            one_data['user'] = uid
            user_ep = user_explicit_preference[uid]
            user_vi = user_vague_intention[uid]["querys"]
            one_data["explicit_preferences"] = user_ep
            one_data["user_related_intention"] = user_vi[0]
            one_data["item_related_intention"] = user_vi[1]

            iid = user_vague_intention[uid]["item"]
            inters = user_vague_intention[uid]["inters"]

            if len(inters) == 0:
                continue

            one_data["item"] = str(iid)

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            one_data["inters"] = ''            
            for item in inters:
                one_data["inters"] = one_data["inters"] + str(item) + ','
            one_data["inters"] = one_data["inters"][:-1]

            search_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(search_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            search_data = np.array(search_data)[sample_idx].tolist()

        return search_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.search_data) * self.prompt_sample_num
        elif self.mode == 'test':
            return len(self.search_data)
        else:
            return len(self.search_data)

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num

        d = self.search_data[idx]
        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))
        d["explicit_preference"] = d["explicit_preference"].replace('{','').replace('}','')
        d["user_related_intention"] = d["user_related_intention"].replace('{','').replace('}','')
        d["item_related_intention"] = d["item_related_intention"].replace('{','').replace('}','')
        all_querys = [d["user_related_intention"], d["item_related_intention"]]
        d["query"] = random.choice(all_querys)

        # d["query"] = d["query"].replace('{','').replace('}','')

        if self.task == 'itemsearch':
            sub_d = d.copy()
            sub_d.pop('inters')
            sub_d.pop('user')
            instruction = prompt["instruction"].format(inters='{inters}', user='{user}', **sub_d)
            input = sft_prompt.format(instruction = instruction, response = "")
            output = sft_prompt.format(instruction = instruction, response = prompt["response"])
            return dict(input_ids = input, labels = output, inters = d['inters'], item = d['item'], user = d['user'], task = self.task, users = 'placeholder')
        elif self.task == 'query2item':
            sub_d = d.copy()
            sub_d.pop('user')
            instruction = prompt["instruction"].format(user='{user}', **sub_d)
            input = sft_prompt.format(instruction = instruction, response = "")
            output = sft_prompt.format(instruction = instruction, response = prompt["response"])
            return dict(input_ids = input, labels = output, inters = 'placeholder', item = d['item'], user = d['user'], task = self.task, users = 'placeholder')

# inters2title & inters2description & intertitles2item
class FusionSeqRecDataset(BaseDataset):
    def __init__(self, args, mode="train", task = 'inters2title',
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.task = task.lower()
        self.prompts = all_prompt[self.task]

        # load data
        self._load_data()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.sample_valid = args.sample_valid
            self.valid_prompt_id = args.valid_prompt_id
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)

    def _process_train_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = str(items[i])
                one_data['user'] = uid
                one_data["title"] = self.item_feat[str(items[i])]["title"].strip().strip(".!?,;:`")
                one_data["title"] = one_data["title"].replace('{','').replace('}','')
                one_data["description"] = self.item_feat[str(items[i])]["description"]
                one_data["description"] = one_data["description"].replace('{','').replace('}','')

                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                
                one_data['inters'] = ''
                for item in history:
                    one_data['inters'] = one_data['inters'] + str(item) +','
                one_data['inters'] = one_data['inters'][:-1]

                inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`").replace('{','').replace('}','') + "\"" for j in history]
                one_data["inter_titles"] = self.his_sep.join(inter_titles)

                inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = str(items[-2])
            one_data["title"] = self.item_feat[str(items[-2])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-2])]["description"]
            one_data["description"] = one_data["description"].replace('{','').replace('}','')

            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data['inters'] = ''
            for item in history:
                one_data['inters'] = one_data['inters'] + str(item) +','
            one_data['inters'] = one_data['inters'][:-1]

            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]
            one_data["inter_titles"] = self.his_sep.join(inter_titles)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]
            one_data = dict()
            one_data["item"] = str(items[-1])
            one_data["title"] = self.item_feat[str(items[-1])]["title"].strip().strip(".!?,;:`")
            one_data["description"] = self.item_feat[str(items[-1])]["description"]

            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]

            one_data['inters'] = ''
            for item in history:
                one_data['inters'] = one_data['inters'] + str(item) +','
            one_data['inters'] = one_data['inters'][:-1]

            inter_titles = ["\"" + self.item_feat[str(j)]["title"].strip().strip(".!?,;:`") + "\"" for j in history]
            one_data["inter_titles"] = self.his_sep.join(inter_titles)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        self.valid_text_data = []
        if self.sample_valid:
            all_prompt_ids = range(len(self.prompts))
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                prompt_ids = np.random.choice(all_prompt_ids, self.prompt_sample_num, replace=False)
                if self.task == 'inters2title':
                    for prompt_id in prompt_ids:
                        prompt = self.prompts[prompt_id]
                        input = sft_prompt.format(instruction = prompt['instruction'], response = "")
                        response = prompt['response'].format(title = d['title'])
                        output = sft_prompt.format(instruction = prompt['instruction'], response = response)
                        self.valid_text_data.append({"input_ids": input, "labels": output, 'inters': d['inters'], 'item': 'placeholder', 'task': self.task})
                elif self.task == 'inters2description':
                    for prompt_id in prompt_ids:
                        prompt = self.prompts[prompt_id]
                        input = sft_prompt.format(instruction = prompt['instruction'], response = "")
                        response = prompt['response'].format(title = d['description'])
                        output = sft_prompt.format(instruction = prompt['instruction'], response = response)
                        self.valid_text_data.append({"input_ids": input, "labels": output, 'inters': d['inters'], 'item': 'placeholder', 'task': self.task})
                elif self.task == 'intertitles2item':
                    for prompt_id in prompt_ids:
                        prompt = self.prompts[prompt_id]
                        instruction = prompt['instruction'].format(inter_titles = d['inter_titles'])
                        input = sft_prompt.format(instruction = instruction, response = "")
                        output = sft_prompt.format(instruction = instruction, response = prompt["response"])
                        self.valid_text_data.append({"input_ids": input, "labels": output, 'inters': 'placeholder', 'item': d['item'], 'task': self.task})
                else:
                    raise NotImplementedError
        else:
            self.prompt_sample_num = 1
            prompt = self.prompts[self.valid_prompt_id]
            for i in range(len(self.inter_data)):
                d = self.inter_data[i]
                if self.task == 'inters2title':
                    input = sft_prompt.format(instruction = prompt['instruction'], response = "")
                    response = prompt['response'].format(title = d['title'])
                    output = sft_prompt.format(instruction = prompt['instruction'], response = response)
                    self.valid_text_data.append({"input_ids": input, "labels": output, 'inters': d['inters'], 'item': 'placeholder', 'task': self.task})
                elif self.task == 'inters2description':
                    input = sft_prompt.format(instruction = prompt['instruction'], response = "")
                    response = prompt['response'].format(title = d['description'])
                    output = sft_prompt.format(instruction = prompt['instruction'], response = response)
                    self.valid_text_data.append({"input_ids": input, "labels": output, 'inters': d['inters'], 'item': 'placeholder', 'task': self.task})
                elif self.task == 'intertitles2item':
                    instruction = prompt['instruction'].format(inter_titles = d['inter_titles'])
                    input = sft_prompt.format(instruction = instruction, response = "")
                    output = sft_prompt.format(instruction = instruction, response = prompt["response"])
                    self.valid_text_data.append({"input_ids": input, "labels": output, 'inters': 'placeholder', 'item': d['item'], 'task': self.task})
                else:
                    raise NotImplementedError

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):
        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]

        if self.mode == 'train':
            prompt_id = random.randint(0, len(self.prompts) - 1)
        elif self.mode == 'test':
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]

        if self.task == 'inters2title':
            input = sft_prompt.format(instruction = prompt['instruction'], response = "")
            response = prompt['response'].format(title = d['title'])
            output = sft_prompt.format(instruction = prompt['instruction'], response = response)
            return dict(input_ids = input, labels = output, inters = d['inters'], user = d['user'], item = 'placeholder', task = self.task, users = 'placeholder')
        elif self.task == 'inters2description':
            input = sft_prompt.format(instruction = prompt['instruction'], response = "")
            response = prompt['response'].format(description = d['description'])
            output = sft_prompt.format(instruction = prompt['instruction'], response = response)
            return dict(input_ids = input, labels = output, inters = d['inters'], user = d['user'], item = 'placeholder', task = self.task, users = 'placeholder')
        elif self.task == 'intertitles2item':
            instruction = prompt['instruction'].format(user = '{user}', inter_titles = d['inter_titles'])
            input = sft_prompt.format(instruction = instruction, response = "")
            output = sft_prompt.format(instruction = instruction, response = prompt["response"])
            return dict(input_ids = input, labels = output, inters = 'placeholder', user = d['user'], item = d['item'], task = self.task, users = 'placeholder')
        else:
            raise NotImplementedError

# preferenceobtain
class PreferenceObtainDataset(BaseDataset):
    def __init__(self, args, prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt["preferenceobtain"]

        # load data
        self._load_data()

        self.preference_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".user.json"), 'r') as f:
            self.user_info = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)

    def _process_data(self):
        preference_data = []
        user_explicit_preference = self.user_info["user_explicit_preference"]

        for uid in user_explicit_preference.keys():
            one_data = {}
            one_data['user'] = uid
            inters = self.inters[uid][:-3]
            user_ep = user_explicit_preference[uid]

            if self.max_his_len > 0:
                inters = inters[-self.max_his_len:]
            one_data['inters'] = ''
            for item in inters:
                one_data['inters'] = one_data['inters'] + str(item) + ','
            one_data['inters'] = one_data['inters'][:-1]

            one_data["explicit_preferences"] = user_ep

            preference_data.append(one_data)

        if self.sample_num > 0:
            all_idx = range(len(preference_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            preference_data = np.array(preference_data)[sample_idx].tolist()

        return preference_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.preference_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):

        idx = index // self.prompt_sample_num

        d = self.preference_data[idx]
        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        d["explicit_preference"] = copy.deepcopy(random.choice(d["explicit_preferences"]))
        d["explicit_preference"] = d["explicit_preference"].replace('{','').replace('}','')

        input = sft_prompt.format(instruction = prompt["instruction"], response = "")
        response = prompt["response"].format(**d)
        output = sft_prompt.format(instruction = prompt["instruction"], response = response)
        return dict(input_ids = input, labels = output, inters = d['inters'], user = d['user'], item = 'placeholder', task = 'preferenceobtain', users = 'placeholder')

# item2index & index2item
class ItemFeatDataset(BaseDataset):
    def __init__(self, args, task="item2index", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.prompts = all_prompt[self.task]

        self._load_data()
        self.feat_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".item.json"), 'r') as f:
            self.item_feat = json.load(f)

    def _process_data(self):
        feat_data = []
        for iid in self.item_feat:
            feat = self.item_feat[iid]
            feat["item"] = iid
            feat["title"] = feat["title"].strip().strip(".!?,;:`")
            feat["title"] = feat["title"].replace('{','').replace('}','')
            feat["description"] = feat["description"].strip().strip(".!?,;:`")
            feat["description"] = feat["description"].replace('{','').replace('}','')
            feat_data.append(feat)

        if self.sample_num > 0:
            all_idx = range(len(feat_data))
            sample_idx = np.random.choice(all_idx, self.sample_num, replace=False)
            feat_data = np.array(feat_data)[sample_idx].tolist()

        return feat_data

    def __len__(self):
        return len(self.feat_data) * self.prompt_sample_num

    def _get_text_data(self, data, prompt):
        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")
        output = sft_prompt.format(instruction = instruction, response = response)

        return input, output

    def __getitem__(self, index):
        idx = index // self.prompt_sample_num
        d = self.feat_data[idx]

        prompt_id = random.randint(0, len(self.prompts) - 1)

        prompt = self.prompts[prompt_id]

        if self.task == 'item2index':
            instruction = prompt["instruction"].format(**d)
            input = sft_prompt.format(instruction = instruction, response = "")
            output = sft_prompt.format(instruction = instruction, response = prompt["response"])
            return dict(input_ids = input, labels = output, inters = 'placeholder', user = 'placeholder', item = d['item'], task = self.task, users = 'placeholder')
        elif self.task == 'index2item':
            input = sft_prompt.format(instruction = prompt["instruction"], response = "")
            response = prompt["response"].format(**d)
            output = sft_prompt.format(instruction = prompt["instruction"], response = response)
            return dict(input_ids = input, labels = output, inters = 'placeholder', user = 'placeholder', item = d['item'], task = self.task, users = 'placeholder')
        else:
            raise NotImplementedError



class SeqRecTestDataset(BaseDataset):

    def __init__(self, args, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompt = all_prompt["seqrec"][self.prompt_id]

        # load data
        self._load_data()
        self._remap_items()

        self.inter_data = self._process_test_data()

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)


    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = self.his_sep.join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)

            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):
        self.prompt_id = prompt_id

        self.prompt = all_prompt["seqrec"][self.prompt_id]

    def __len__(self):

        return len(self.inter_data)

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")

        return input, response

    def __getitem__(self, index):

        d = self.inter_data[index]
        input, target = self._get_text_data(d, self.prompt)

        return dict(input_ids=input, labels=target)