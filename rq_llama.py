import os
import json
import copy
import wandb
import torch
import torch.nn as nn
import transformers
from transformers import LlamaPreTrainedModel, LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from index.models import *
from index.models.rqvae import RQVAE

def _similarity(self, h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

def infonce(self, anchor, sample, tau=0.1):
    sim = self._similarity(anchor, sample) / tau
    num_nodes = anchor.shape[0]
    device = anchor.device

    factor = torch.tensor([[1.]]).to(device)
    scalor = self.projector(factor)
    sim = sim * scalor

    pos_mask = torch.eye(num_nodes, dtype=torch.float32).to(device)
    neg_mask = 1. - pos_mask
    assert sim.size() == pos_mask.size()  # sanity check

    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
    return -loss.mean()

class LlamaWithRQ(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        args = config.args
        
        tokenizer = LlamaTokenizer.from_pretrained(
            args['base_model'],
            model_max_length = args['model_max_length'],
            padding_side ="right",
        )
        tokenizer.pad_token_id = 0

        item_tokens = []
        prefix = ['<a-{}>','<b-{}>','<c-{}>','<d-{}>','<e-{}>']
        for i in range(len(args['num_emb_list'])):
            item_tokens.extend([prefix[i].format(int(x)) for x in range(args['num_emb_list'][i])])
        self.prefix = prefix

        user_tokens = []
        user_prefix = ['<z-{}>','<y-{}>','<x-{}>','<w-{}>','<v-{}>']
        for i in range(len(args['num_emb_list'])):
            user_tokens.extend([user_prefix[i].format(int(x)) for x in range(args['num_emb_list'][i])])
        self.user_prefix = user_prefix

        tokenizer.add_tokens(item_tokens)
        tokenizer.add_tokens(user_tokens)
        config.vocab_size = len(tokenizer)

        llama_model = LlamaForCausalLM.from_pretrained(args['base_model'])
        llama_model.resize_token_embeddings(len(tokenizer))
        
        lora_config = LoraConfig(
            r = args['lora_r'],
            lora_alpha = args['lora_alpha'],
            target_modules = args['lora_target_modules'].split(","),
            modules_to_save = args['lora_modules_to_save'].split(","),
            lora_dropout = args['lora_dropout'],
            bias = "none",
            inference_mode = False,
            task_type = TaskType.CAUSAL_LM
        )
        llama_model = get_peft_model(llama_model, lora_config)

        for n, p in llama_model.named_parameters():
            if "original_module" in n and any(module_name in n for module_name in lora_config.modules_to_save):
                p.requires_grad = False

        self.tokenizer = tokenizer
        self.model = llama_model

        item_json = os.path.join(args['data_path'], args['dataset'], args['dataset'] + ".item.json")
        with open(item_json, 'r') as f:
            self.item_texts = json.load(f)

        user_json = os.path.join(args['data_path'], args['dataset'], args['dataset'] + ".user.json")
        with open(user_json, 'r') as f:
            self.user_texts = json.load(f)
            self.user_texts = self.user_texts['user_explicit_preference']

        self.item_rqvae = RQVAE(
            in_dim = config.hidden_size,
            num_emb_list = args['num_emb_list'],
            e_dim = args['e_dim'],
            layers = args['layers'],
            dropout_prob = args['dropout_prob'],
            bn = args['bn'],
            loss_type = args['loss_type'],
            quant_loss_weight = args['quant_loss_weight'],
            kmeans_init = args['kmeans_init'],
            kmeans_iters = args['kmeans_iters'],
            sk_epsilons = args['sk_epsilons'],
            sk_iters = args['sk_iters'])

        self.user_rqvae = RQVAE(
            in_dim = config.hidden_size,
            num_emb_list = args['num_emb_list'],
            e_dim = args['e_dim'],
            layers = args['layers'],
            dropout_prob = args['dropout_prob'],
            bn = args['bn'],
            loss_type = args['loss_type'],
            quant_loss_weight = args['quant_loss_weight'],
            kmeans_init = args['kmeans_init'],
            kmeans_iters = args['kmeans_iters'],
            sk_epsilons = args['sk_epsilons'],
            sk_iters = args['sk_iters'])

        # self.projector = nn.Linear(args['e_dim'], config.hidden_size)
        # self.item_projector = nn.Linear(args['e_dim'], config.hidden_size)
        # self.user_projector = nn.Linear(args['e_dim'], config.hidden_size)
        self.args = args

    def rqvae_forward(self, inputs, targets, inters, item, users, user, task):
        llama_model = self.model.get_decoder()
        if task.lower() == 'seqrec':
            # inters, user, item

            inter_feature_list = []
            inter_emb_list = []
            inter_item_list = inters.split(',')
            for j in range(len(inter_item_list)):
                inter_feature = self.item_texts[inter_item_list[j]]['title'] + ' ' + self.item_texts[inter_item_list[j]]['description']
                inter_id = self.tokenizer(inter_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
                inter_emb = llama_model(input_ids = inter_id.input_ids, attention_mask = inter_id.attention_mask)
                inter_emb = inter_emb.last_hidden_state * inter_id.attention_mask.unsqueeze(-1)
                inter_emb = inter_emb.sum(dim=1) / inter_id.attention_mask.sum(dim = -1, keepdim = True)
                inter_emb_list.append(inter_emb.detach())
            inter_embs = torch.cat(inter_emb_list, dim = 0)

            item_feature = self.item_texts[item]['title'] + ' ' + self.item_texts[item]['description']
            item_ids = self.tokenizer(item_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
            item_emb = llama_model(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
            item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
            item_emb = item_emb.sum(dim=1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)
            item_emb = item_emb.detach()

            user_feature = " ".join(self.user_texts[user])
            user_ids = self.tokenizer(user_feature, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_emb = llama_model(input_ids = user_ids.input_ids, attention_mask = user_ids.attention_mask)
            user_emb = user_emb.last_hidden_state * user_ids.attention_mask.unsqueeze(-1)
            user_emb = user_emb.sum(dim=1) / user_ids.attention_mask.sum(dim = -1, keepdim = True)
            user_emb = user_emb.detach()

            item_rec_embs, item_rq_loss, item_rqids = self.item_rqvae(torch.cat([inter_embs, item_emb], dim = 0))
            item_rqvae_loss, item_rec_loss = self.item_rqvae.compute_loss(item_rec_embs, item_rq_loss, torch.cat([inter_embs, item_emb], dim = 0))
            
            user_rec_emb, user_rq_loss, user_rqids = self.user_rqvae(user_emb)
            user_rqvae_loss, user_rec_loss = self.user_rqvae.compute_loss(user_rec_emb, user_rq_loss, user_emb)

            inters_rqids = item_rqids.view(-1, item_rqids.shape[-1]).cpu().numpy().tolist()[:-1]
            item_rqid = item_rqids.view(-1, item_rqids.shape[-1]).cpu().numpy().tolist()[-1]
            user_rqid = user_rqids.view(-1, user_rqids.shape[-1]).cpu().numpy().tolist()[0]
                
            text_rqids = {}
            item_sid_list = []

            code = ''
            for rqid in inters_rqids:
                item_sid = ''
                for k, idx in enumerate(rqid):
                    item_sid = item_sid + self.prefix[k].format(idx)
                    code = code + self.prefix[k].format(idx)
                code = code + ', '
                item_sid_list.append(item_sid)
            text_rqids['inters'] = code[:-2]

            code = ''
            for k, idx in enumerate(item_rqid):
                code = code + self.prefix[k].format(idx)
            item_sid_list.append(code)
            text_rqids['item'] = code

            code = ''
            for k, idx in enumerate(user_rqid):
                code = code + self.user_prefix[k].format(idx)
            text_rqids['user'] = code
            user_sid_list = [code]

            item_sid_ids = self.tokenizer(item_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            item_sid_emb = llama_model(input_ids = item_sid_ids.input_ids, attention_mask = item_sid_ids.attention_mask)
            item_sid_emb = item_sid_emb.last_hidden_state * item_sid_ids.attention_mask.unsqueeze(-1)
            item_sid_emb = item_sid_emb.sum(dim = 1) / item_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            cts_align_loss = infonce(item_sid_emb, torch.cat([inter_embs, item_emb], dim = 0))
            num_cts = item_rec_embs.shape[0]

            user_sid_ids = self.tokenizer(user_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_sid_emb = llama_model(input_ids = user_sid_ids.input_ids, attention_mask = user_sid_ids.attention_mask)
            user_sid_emb = user_sid_emb.last_hidden_state * user_sid_ids.attention_mask.unsqueeze(-1)
            user_sid_emb = user_sid_emb.sum(dim = 1) / user_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            rec_align_loss = F.mse_loss(user_sid_emb, user_emb, reduction = 'mean')
            num_rec = user_rec_emb.shape[0]

            inputs = inputs.format(inters = text_rqids['inters'], user = text_rqids['user'])
            targets = targets.format(inters = text_rqids['inters'], user = text_rqids['user'], item = text_rqids['item'])

            num_item = item_rec_embs.shape[0]
            num_user = user_rec_emb.shape[0]

        elif task.lower() == 'itemsearch':
            # inters, item
            inter_feature_list = []
            inter_emb_list = []
            inter_item_list = inters.split(',')
            for j in range(len(inter_item_list)):
                inter_feature = self.item_texts[inter_item_list[j]]['title'] + ' ' + self.item_texts[inter_item_list[j]]['description']
                inter_id = self.tokenizer(inter_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
                inter_emb = llama_model(input_ids = inter_id.input_ids, attention_mask = inter_id.attention_mask)
                inter_emb = inter_emb.last_hidden_state * inter_id.attention_mask.unsqueeze(-1)
                inter_emb = inter_emb.sum(dim=1) / inter_id.attention_mask.sum(dim = -1, keepdim = True)
                inter_emb_list.append(inter_emb.detach())
            inter_embs = torch.cat(inter_emb_list, dim = 0)

            item_feature = self.item_texts[item]['title'] + ' ' + self.item_texts[item]['description']
            item_ids = self.tokenizer(item_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
            item_emb = llama_model(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
            item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
            item_emb = item_emb.sum(dim=1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)
            item_emb = item_emb.detach()

            # user_feature = " ".join(self.user_texts[user])
            # user_ids = self.tokenizer(user_feature, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            # user_emb = llama_model(input_ids = user_ids.input_ids, attention_mask = user_ids.attention_mask)
            # user_emb = user_emb.last_hidden_state * user_ids.attention_mask.unsqueeze(-1)
            # user_emb = user_emb.sum(dim=1) / user_ids.attention_mask.sum(dim = -1, keepdim = True)
            # user_emb = user_emb.detach()

            item_rec_embs, item_rq_loss, item_rqids = self.item_rqvae(torch.cat([inter_embs, item_emb], dim = 0))
            item_rqvae_loss, item_rec_loss = self.item_rqvae.compute_loss(item_rec_embs, item_rq_loss, torch.cat([inter_embs, item_emb], dim = 0))
            
            # user_rec_emb, user_rq_loss, user_rqids = self.user_rqvae(user_emb)
            # user_rqvae_loss, user_rec_loss = self.user_rqvae.compute_loss(user_rec_emb, user_rq_loss, user_emb)

            inters_rqids = item_rqids.view(-1, item_rqids.shape[-1]).cpu().numpy().tolist()[:-1]
            item_rqid = item_rqids.view(-1, item_rqids.shape[-1]).cpu().numpy().tolist()[-1]
            # user_rqid = user_rqids.view(-1, user_rqids.shape[-1]).cpu().numpy().tolist()[0]
                
            text_rqids = {}
            item_sid_list = []

            code = ''
            for rqid in inters_rqids:
                item_sid = ''
                for k, idx in enumerate(rqid):
                    item_sid = item_sid + self.prefix[k].format(idx)
                    code = code + self.prefix[k].format(idx)
                code = code + ', '
                item_sid_list.append(item_sid)
            text_rqids['inters'] = code[:-2]

            code = ''
            for k, idx in enumerate(item_rqid):
                code = code + self.prefix[k].format(idx)
            item_sid_list.append(code)
            text_rqids['item'] = code
            # code = ''
            # for k, idx in enumerate(user_rqid):
            #     code = code + self.user_prefix[k].format(idx)
            # text_rqids['user'] = code

            item_sid_ids = self.tokenizer(item_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            item_sid_emb = llama_model(input_ids = item_sid_ids.input_ids, attention_mask = item_sid_ids.attention_mask)
            item_sid_emb = item_sid_emb.last_hidden_state * item_sid_ids.attention_mask.unsqueeze(-1)
            item_sid_emb = item_sid_emb.sum(dim = 1) / item_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            cts_align_loss = infonce(item_sid_emb, torch.cat([inter_embs, item_emb], dim = 0))
            num_cts = item_rec_embs.shape[0]

            rec_align_loss, num_rec = 0, 0

            inputs = inputs.format(inters = text_rqids['inters'])
            targets = targets.format(inters = text_rqids['inters'], item = text_rqids['item'])

            num_item = item_rec_embs.shape[0]
            num_user = 0
            user_rqvae_loss = 0

        elif task.lower() in ['inters2title','inters2description']:
            # inputs, targets, inters, user
            inter_feature_list = []
            inter_emb_list = []
            inter_item_list = inters.split(',')
            for j in range(len(inter_item_list)):
                inter_feature = self.item_texts[inter_item_list[j]]['title'] + ' ' + self.item_texts[inter_item_list[j]]['description']
                inter_id = self.tokenizer(inter_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
                inter_emb = llama_model(input_ids = inter_id.input_ids, attention_mask = inter_id.attention_mask)
                inter_emb = inter_emb.last_hidden_state * inter_id.attention_mask.unsqueeze(-1)
                inter_emb = inter_emb.sum(dim=1) / inter_id.attention_mask.sum(dim = -1, keepdim = True)
                inter_emb_list.append(inter_emb.detach())
            inter_embs = torch.cat(inter_emb_list, dim = 0)

            user_feature = " ".join(self.user_texts[user])
            user_ids = self.tokenizer(user_feature, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_emb = llama_model(input_ids = user_ids.input_ids, attention_mask = user_ids.attention_mask)
            user_emb = user_emb.last_hidden_state * user_ids.attention_mask.unsqueeze(-1)
            user_emb = user_emb.sum(dim=1) / user_ids.attention_mask.sum(dim = -1, keepdim = True)
            user_emb = user_emb.detach()

            item_rec_embs, item_rq_loss, item_rqids = self.item_rqvae(inter_embs)
            item_rqvae_loss, item_rec_loss = self.item_rqvae.compute_loss(item_rec_embs, item_rq_loss, inter_embs)

            user_rec_emb, user_rq_loss, user_rqids = self.user_rqvae(user_emb)
            user_rqvae_loss, user_rec_loss = self.user_rqvae.compute_loss(user_rec_emb, user_rq_loss, user_emb)

            inters_rqids = item_rqids.view(-1, item_rqids.shape[-1]).cpu().numpy().tolist()
            user_rqid = user_rqids.view(-1, user_rqids.shape[-1]).cpu().numpy().tolist()[0]
            
            text_rqids = {}
            item_sid_list = []

            code = ''
            for rqid in inters_rqids:
                item_sid = ''
                for k, idx in enumerate(rqid):
                    item_sid = item_sid + self.prefix[k].format(idx)
                    code = code + self.prefix[k].format(idx)
                code = code + ', '
            item_sid_list.append(item_sid)
            text_rqids['inters'] = code[:-2]
            
            code = ''
            for k, idx in enumerate(user_rqid):
                code = code + self.user_prefix[k].format(idx)
            text_rqids['user'] = code
            user_sid_list = [code]

            item_sid_ids = self.tokenizer(item_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            item_sid_emb = llama_model(input_ids = item_sid_ids.input_ids, attention_mask = item_sid_ids.attention_mask)
            item_sid_emb = item_sid_emb.last_hidden_state * item_sid_ids.attention_mask.unsqueeze(-1)
            item_sid_emb = item_sid_emb.sum(dim = 1) / item_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            cts_align_loss = infonce(item_sid_emb, inter_embs)
            num_cts = item_rec_embs.shape[0]

            user_sid_ids = self.tokenizer(user_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_sid_emb = llama_model(input_ids = user_sid_ids.input_ids, attention_mask = user_sid_ids.attention_mask)
            user_sid_emb = user_sid_emb.last_hidden_state * user_sid_ids.attention_mask.unsqueeze(-1)
            user_sid_emb = user_sid_emb.sum(dim = 1) / user_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            rec_align_loss += F.mse_loss(user_sid_emb, user_emb, reduction = 'mean')
            num_rec = user_rec_emb.shape[0]

            inputs = inputs.format(inters = text_rqids['inters'], user = text_rqids['user'])
            targets = targets.format(inters = text_rqids['inters'], user = text_rqids['user'])

            num_item = item_rec_embs.shape[0]
            num_user = user_rec_emb.shape[0]

        elif task.lower() in ['intertitles2item','query2item']:
            # inputs, targets, item, user
            item_feature = self.item_texts[item]['title'] + ' ' + self.item_texts[item]['description']
            item_ids = self.tokenizer(item_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
            item_emb = llama_model(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
            item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
            item_emb = item_emb.sum(dim=1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)
            item_emb = item_emb.detach()

            user_feature = " ".join(self.user_texts[user])
            user_ids = self.tokenizer(user_feature, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_emb = llama_model(input_ids = user_ids.input_ids, attention_mask = user_ids.attention_mask)
            user_emb = user_emb.last_hidden_state * user_ids.attention_mask.unsqueeze(-1)
            user_emb = user_emb.sum(dim=1) / user_ids.attention_mask.sum(dim = -1, keepdim = True)
            user_emb = user_emb.detach()

            item_rec_embs, item_rq_loss, item_rqids = self.item_rqvae(item_emb)
            item_rqvae_loss, item_rec_loss = self.item_rqvae.compute_loss(item_rec_embs, item_rq_loss, item_emb)

            user_rec_emb, user_rq_loss, user_rqids = self.user_rqvae(user_emb)
            user_rqvae_loss, user_rec_loss = self.user_rqvae.compute_loss(user_rec_emb, user_rq_loss, user_emb)

            item_rqid = item_rqids.view(-1, item_rqids.shape[-1]).cpu().numpy().tolist()[0]
            user_rqid = user_rqids.view(-1, user_rqids.shape[-1]).cpu().numpy().tolist()[0]

            text_rqids = {}
            code = ''
            for k, idx in enumerate(item_rqid):
                code = code + self.prefix[k].format(idx)
            text_rqids['item'] = code
            item_sid_list = [code]

            code = ''
            for k, idx in enumerate(user_rqid):
                code = code + self.user_prefix[k].format(idx)
            text_rqids['user'] = code
            user_sid_list = [code]

            item_sid_ids = self.tokenizer(item_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            item_sid_emb = llama_model(input_ids = item_sid_ids.input_ids, attention_mask = item_sid_ids.attention_mask)
            item_sid_emb = item_sid_emb.last_hidden_state * item_sid_ids.attention_mask.unsqueeze(-1)
            item_sid_emb = item_sid_emb.sum(dim = 1) / item_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            rec_align_loss = F.mse_loss(item_sid_emb, item_emb, reduction = 'mean')

            user_sid_ids = self.tokenizer(user_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_sid_emb = llama_model(input_ids = user_sid_ids.input_ids, attention_mask = user_sid_ids.attention_mask)
            user_sid_emb = user_sid_emb.last_hidden_state * user_sid_ids.attention_mask.unsqueeze(-1)
            user_sid_emb = user_sid_emb.sum(dim = 1) / user_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            rec_align_loss += F.mse_loss(user_sid_emb, user_emb, reduction = 'mean')
            num_rec = item_rec_embs.shape[0] + user_rec_emb.shape[0]

            cts_align_loss, num_cts = 0, 0
            
            inputs = inputs.format(user = text_rqids['user'])
            targets = targets.format(item = text_rqids['item'], user = text_rqids['user'])

            num_item = item_rec_embs.shape[0]
            num_user = user_rec_emb.shape[0]

        elif task.lower() in ['item2index','index2item']:
            # inputs, targets, item
            item_feature = self.item_texts[item]['title'] + ' ' + self.item_texts[item]['description']
            item_ids = self.tokenizer(item_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
            item_emb = llama_model(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
            item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
            item_emb = item_emb.sum(dim=1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)
            item_emb = item_emb.detach()

            rec_embs, rq_loss, rqids = self.item_rqvae(item_emb)
            rqvae_loss, rec_loss = self.item_rqvae.compute_loss(rec_embs, rq_loss, item_emb)

            item_rqid = rqids.view(-1, rqids.shape[-1]).cpu().numpy().tolist()[0]
            code = ''
            for k, idx in enumerate(item_rqid):
                code = code + self.prefix[k].format(idx)
            item_sid_list = [code]

            item_sid_ids = self.tokenizer(item_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            item_sid_emb = llama_model(input_ids = item_sid_ids.input_ids, attention_mask = item_sid_ids.attention_mask)
            item_sid_emb = item_sid_emb.last_hidden_state * item_sid_ids.attention_mask.unsqueeze(-1)
            item_sid_emb = item_sid_emb.sum(dim = 1) / item_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            rec_align_loss = F.mse_loss(item_sid_emb, item_emb, reduction = 'mean')
            num_rec = rec_embs.shape[0]

            cts_align_loss, num_cts = 0, 0

            if task.lower() == 'item2index':
                targets = targets.format(item = code)
            elif task.lower() == 'index2item':
                inputs = inputs.format(item = code)
                targets = targets.format(item = code)
            else:
                raise NotImplementedError

            item_rqvae_loss = rqvae_loss
            user_rqvae_loss = 0
            num_item = rec_embs.shape[0]
            num_user = 0

        elif task.lower() == 'preferenceobtain':
            # inputs, targets, inters
            inter_feature_list = []
            inter_emb_list = []
            inter_item_list = inters.split(',')
            for j in range(len(inter_item_list)):
                inter_feature = self.item_texts[inter_item_list[j]]['title'] + ' ' + self.item_texts[inter_item_list[j]]['description']
                inter_id = self.tokenizer(inter_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
                inter_emb = llama_model(input_ids = inter_id.input_ids, attention_mask = inter_id.attention_mask)
                inter_emb = inter_emb.last_hidden_state * inter_id.attention_mask.unsqueeze(-1)
                inter_emb = inter_emb.sum(dim=1) / inter_id.attention_mask.sum(dim = -1, keepdim = True)
                inter_emb_list.append(inter_emb.detach())
            inter_embs = torch.cat(inter_emb_list, dim = 0)

            rec_embs, rq_loss, rqids = self.item_rqvae(inter_embs)
            rqvae_loss, rec_loss = self.item_rqvae.compute_loss(rec_embs, rq_loss, inter_embs)

            inters_rqids = rqids.view(-1, rqids.shape[-1]).cpu().numpy().tolist()
            
            item_sid_list = []
            code = ''
            for rqid in inters_rqids:
                item_sid = ''
                for k, idx in enumerate(rqid):
                    item_sid = item_sid + self.prefix[k].format(idx)
                    code = code + self.prefix[k].format(idx)
                code = code + ', '
                item_sid_list.append(item_sid)
            code = code[:-2]

            item_sid_ids = self.tokenizer(item_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            item_sid_emb = llama_model(input_ids = item_sid_ids.input_ids, attention_mask = item_sid_ids.attention_mask)
            item_sid_emb = item_sid_emb.last_hidden_state * item_sid_ids.attention_mask.unsqueeze(-1)
            item_sid_emb = item_sid_emb.sum(dim = 1) / item_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            cts_align_loss = infonce(item_sid_emb, inter_embs)
            num_cts = rec_embs.shape[0]

            rec_align_loss, num_rec = 0, 0

            inputs = inputs.format(inters = code)
            targets = targets.format(inters = code)

            item_rqvae_loss = rqvae_loss
            user_rqvae_loss = 0
            num_item = rec_embs.shape[0]
            num_user = 0

        elif task.lower() == 'usersearch':
            # item, users, user
            item_feature = self.item_texts[item]['title'] + ' ' + self.item_texts[item]['description']
            item_ids = self.tokenizer(item_feature, return_tensors = 'pt', padding=True, truncation=True).to(self.model.device)
            item_emb = llama_model(input_ids = item_ids.input_ids, attention_mask = item_ids.attention_mask)
            item_emb = item_emb.last_hidden_state * item_ids.attention_mask.unsqueeze(-1)
            item_emb = item_emb.sum(dim = 1) / item_ids.attention_mask.sum(dim = -1, keepdim = True)
            item_emb = item_emb.detach()

            users_emb_list = []
            users_list = users.split(',')
            for j in range(len(users_list)):
                u_feature = " ".join(self.user_texts[users_list[j]])
                u_id = self.tokenizer(u_feature, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
                u_emb = llama_model(input_ids = u_id.input_ids, attention_mask = u_id.attention_mask)
                u_emb = u_emb.last_hidden_state * u_id.attention_mask.unsqueeze(-1)
                u_emb = u_emb.sum(dim = 1) / u_id.attention_mask.sum(dim = -1, keepdim = True)
                users_emb_list.append(u_emb.detach())
            users_emb = torch.cat(users_emb_list, dim = 0)

            user_feature = " ".join(self.user_texts[user])
            user_ids = self.tokenizer(user_feature, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_emb = llama_model(input_ids = user_ids.input_ids, attention_mask = user_ids.attention_mask)
            user_emb = user_emb.last_hidden_state * user_ids.attention_mask.unsqueeze(-1)
            user_emb = user_emb.sum(dim = 1) / user_ids.attention_mask.sum(dim = -1, keepdim = True)
            user_emb = user_emb.detach()

            item_rec_embs, item_rq_loss, item_rqids = self.item_rqvae(item_emb)
            item_rqvae_loss, item_rec_loss = self.item_rqvae.compute_loss(item_rec_embs, item_rq_loss, item_emb)

            user_rec_emb, user_rq_loss, user_rqids = self.user_rqvae(torch.cat([users_emb, user_emb], dim = 0))
            user_rqvae_loss, user_rec_loss = self.user_rqvae.compute_loss(user_rec_emb, user_rq_loss, torch.cat([users_emb, user_emb], dim = 0))

            item_rqid = item_rqids.view(-1, item_rqids.shape[-1]).cpu().numpy().tolist()[0]
            users_rqids = user_rqids.view(-1, user_rqids.shape[-1]).cpu().numpy().tolist()[:-1]
            user_rqid = user_rqids.view(-1, user_rqids.shape[-1]).cpu().numpy().tolist()[-1]

            text_rqids = {}
            code = ''
            for k, idx in enumerate(item_rqid):
                code = code + self.prefix[k].format(idx)
            text_rqids['item'] = code
            item_sid_list = [code]
            
            user_item_list = []
            code = ''
            for rqid in users_rqids:
                user_sid = ''
                for k, idx in enumerate(rqid):
                    user_sid += self.user_prefix[k].format(idx)
                    code = code + self.user_prefix[k].format(idx)
                code = code + ', '
                user_item_list.append(user_sid)
            text_rqids['users'] = code[:-2]

            code = ''
            for k, idx in enumerate(user_rqid):
                code = code + self.user_prefix[k].format(idx)
            text_rqids['user'] = code
            user_sid_list.append(code)

            item_sid_ids = self.tokenizer(item_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            item_sid_emb = llama_model(input_ids = item_sid_ids.input_ids, attention_mask = item_sid_ids.attention_mask)
            item_sid_emb = item_sid_emb.last_hidden_state * item_sid_ids.attention_mask.unsqueeze(-1)
            item_sid_emb = item_sid_emb.sum(dim = 1) / item_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            rec_align_loss = F.mse_loss(item_sid_emb, item_emb, reduction = 'mean')
            num_rec = item_rec_embs.shape[0]

            user_sid_ids = self.tokenizer(user_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_sid_emb = llama_model(input_ids = user_sid_ids.input_ids, attention_mask = user_sid_ids.attention_mask)
            user_sid_emb = user_sid_emb.last_hidden_state * user_sid_ids.attention_mask.unsqueeze(-1)
            user_sid_emb = user_sid_emb.sum(dim = 1) / user_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            cts_align_loss += infonce(user_sid_emb, user_emb)
            num_cts = user_rec_emb.shape[0]

            inputs = inputs.format(item = text_rqids['item'], users = text_rqids['users'])
            targets = targets.format(item = text_rqids['item'], users = text_rqids['users'], user = text_rqids['user'])

            num_item = item_rec_embs.shape[0]
            num_user = user_rec_emb.shape[0]

        elif task.lower() in ['pref2user','user2pref']:
            # inputs, targets, user
            user_feature = " ".join(self.user_texts[user])
            user_ids = self.tokenizer(user_feature, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_emb = llama_model(input_ids = user_ids.input_ids, attention_mask = user_ids.attention_mask)
            user_emb = user_emb.last_hidden_state * user_ids.attention_mask.unsqueeze(-1)
            user_emb = user_emb.sum(dim = 1) / user_ids.attention_mask.sum(dim = -1, keepdim = True)
            user_emb = user_emb.detach()

            user_rec_emb, user_rq_loss, user_rqids = self.user_rqvae(user_emb)
            user_rqvae_loss, user_rec_loss = self.user_rqvae.compute_loss(user_rec_emb, user_rq_loss, user_emb)

            user_rqid = user_rqids.view(-1, user_rqids.shape[-1]).cpu().numpy().tolist()[0]
            code = ''
            for k, idx in enumerate(user_rqid):
                code = code + self.user_prefix[k].format(idx)
            user_sid_list = [code]

            user_sid_ids = self.tokenizer(user_sid_list, return_tensors = 'pt', padding = True, truncation = True).to(self.model.device)
            user_sid_emb = llama_model(input_ids = user_sid_ids.input_ids, attention_mask = user_sid_ids.attention_mask)
            user_sid_emb = user_sid_emb.last_hidden_state * user_sid_ids.attention_mask.unsqueeze(-1)
            user_sid_emb = user_sid_emb.sum(dim = 1) / user_sid_ids.attention_mask.sum(dim = -1, keepdim = True)
            rec_align_loss = F.mse_loss(user_sid_emb, user_emb, reduction = 'mean')
            num_rec = user_rec_emb.shape[0]

            cts_align_loss, num_cts = 0, 0

            if task.lower() == 'pref2user':
                targets = targets.format(user = code)
            elif task.lower() == 'user2pref':
                inputs = inputs.format(user = code)
                targets = targets.format(user = code)
            else:
                raise NotImplementedError

            item_rqvae_loss = 0
            num_item = 0
            num_user = user_rec_emb.shape[0]

        else:
            raise NotImplementedError
    
        return inputs, targets, item_rqvae_loss, user_rqvae_loss, num_item, num_user, cts_align_loss, rec_align_loss, num_cts, num_rec

    def forward(self, input_ids, labels, inters, item, users, user, task):
        '''
        'input_ids': 
        [
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
                Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. 
                The historical interactions are provided as follows: {inters}.
            ### Response:", 
            
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
                You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. 
                Using this history as a reference, please select the next item to recommend to the user.
            ### Response:'
        ], 
        
        'labels': 
        [
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
                Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. 
                The historical interactions are provided as follows: {inters}.
            ### Response:{item}", 
            
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
                You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. 
                Using this history as a reference, please select the next item to recommend to the user.
            ### Response:{item}'
        ], 
        
        'inters': ['0', '0,1'], 
        'item': ['1', '2'], 
        'task': ['seqrec', 'seqrec']
        '''
        assert len(set([len(input_ids), len(labels), len(inters), len(item), len(user), len(task), len(users)])) == 1
        num_data = len(task)

        total_item_rqvae_loss, total_user_rqvae_loss, total_cts_align_loss, total_rec_align_loss = 0, 0, 0, 0
        total_num_item, total_num_user, total_num_cts, total_num_rec = 1e-8, 1e-8, 1e-8, 1e-8
        for i in range(num_data):
            input_ids[i], labels[i], item_rqvae_loss, user_rqvae_loss, num_item, num_user, cts_align_loss, rec_align_loss, num_cts, num_rec = self.rqvae_forward(
                input_ids[i], labels[i], inters[i], item[i], users[i], user[i], task[i])
            total_item_rqvae_loss = total_item_rqvae_loss + item_rqvae_loss * num_item
            total_user_rqvae_loss = total_user_rqvae_loss + user_rqvae_loss * num_user
            total_cts_align_loss = total_cts_align_loss + cts_align_loss * num_cts
            total_rec_align_loss = total_rec_align_loss + rec_align_loss * num_rec
            total_num_item += num_item
            total_num_user += num_user
            total_num_cts += num_cts
            total_num_rec += num_rec
        
        input_data = self.tokenizer(
            text = labels,
            text_target = input_ids,
            return_tensors = 'pt',
            padding = 'longest',
            truncation = True,
            max_length = self.tokenizer.model_max_length,
            return_attention_mask = True
        ).to(self.model.device)

        labels = copy.deepcopy(input_data["input_ids"])
        if self.args['only_train_response']:
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[torch.where(input_data["labels"] != self.tokenizer.pad_token_id)] = -100

        input_data["labels"] = labels

        # codebook_embedding = []
        # for i in range(len(self.item_rqvae.num_emb_list)):
        #     codebook_embedding.append(self.item_rqvae.rq.vq_layers[i].embedding.weight.data)
        # for i in range(len(self.user_rqvae.num_emb_list)):
        #     codebook_embedding.append(self.user_rqvae.rq.vq_layers[i].embedding.weight.data)
        # codebook_embedding = torch.cat(codebook_embedding, dim = 0)
        # codebook_embedding = self.projector(codebook_embedding)
        # self.model.model.model.embed_tokens.weight.data[-codebook_embedding.shape[0]:] = codebook_embedding

        # input_data: dict_keys(['input_ids', 'attention_mask', 'labels'])
        result = self.model(**input_data)
        # wandb.log({'Llama_Loss': result.loss, 'RQVAE_Loss': total_rqvae_loss / total_num_sample})
        result.loss = result.loss +  total_item_rqvae_loss / total_num_item + total_user_rqvae_loss / total_num_user
        result.loss = result.loss +  total_cts_align_loss / total_num_cts + total_rec_align_loss / total_num_rec
        # wandb.log({'Total_Loss': result.loss})
        return result

    def floating_point_ops(self, inputs):
        return 0