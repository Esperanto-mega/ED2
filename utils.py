import json
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data import *
# from data import SeqRecDataset, ItemFeatDataset, ItemSearchDataset, FusionSeqRecDataset, SeqRecTestDataset, PreferenceObtainDataset
from data_finetune import *
# from data_finetune import SeqRecFinetune, ItemFeatFinetune, ItemSearchFinetune, FusionSeqRecFinetune, PreferenceObtainFinetune

def parse_evaluate_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="../llama-7b/", help="basic model path")
    parser.add_argument("--output_dir", type=str, default="./ckpt/", help="The output directory")

    parser.add_argument("--data_path", type=str, default="",
                        help="data directory")
    parser.add_argument("--tasks", type=str, 
                        default='seqrec,itemsearch,inters2title,inters2description,preferenceobtain,item2index,index2item,intertitles2item,query2item',
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--train_data_sample_num", type=str, default="0,0,0,0,0,0,0,0,0",
                        help="the number of sampling data for each task")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.item.json", help="the item indices file")
    parser.add_argument("--user_index_file", type=str, default=".index.user.json", help="the item indices file")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="dataloader prefetch_factor")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1,1,1,1,1,1,1,1,1",
                        help="the number of sampling prompts for each task")

    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")

    parser.add_argument("--ckpt_path", type=str, default="", help="The checkpoint path")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str, default="./results/test-ddp.json", help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")

    return parser

def parse_finetune_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="../llama-7b/", help="basic model path")
    
    parser.add_argument("--output_dir", type=str, default="./ckpt/", help="The output directory")

    parser.add_argument("--data_path", type=str, default="",
                        help="data directory")
    parser.add_argument("--tasks", type=str, 
                        default='seqrec,itemsearch,inters2title,inters2description,preferenceobtain,item2index,index2item,intertitles2item,query2item',
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--train_data_sample_num", type=str, default="0,0,0,0,0,0,0,0,0",
                        help="the number of sampling data for each task")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="item indices file")
    parser.add_argument("--user_index_file", type=str, default=".user-index.json", help="user indices file")
    
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="dataloader prefetch_factor")

    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1,1,1,1,1,1,1,1,1",
                        help="the number of sampling prompts for each task")

    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")

    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj", help="separate by comma")
    parser.add_argument("--lora_modules_to_save", type=str,
                        default="embed_tokens,lm_head", help="separate by comma")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--remove_unused_columns", action="store_true", default=False, help='if remove unused columns')

    parser.add_argument("--reindex", type = int, default = 0)
    # parser.add_argument("--user_reindex", type = int, default = 0)
    parser.add_argument("--ckpt_path", type=str, default="")

    return parser

def load_finetune_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecFinetune(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "item2index" or task.lower() == "index2item":
            dataset = ItemFeatFinetune(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "fusionseqrec":
            dataset = FusionSeqRecFinetune(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "itemsearch":
            dataset = ItemSearchFinetune(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "preferenceobtain":
            dataset = PreferenceObtainFinetune(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
        
        elif task.lower() == "usersearch":
            dataset = UserSearchFinetune(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() in ["user2pref", "pref2user"]:
            dataset = UserFeatFinetune(args, task = task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecFinetune(args, "valid", args.valid_prompt_sample_num)

    return train_data, valid_data

# def load_finetune_datasets(args):
#     tasks = args.tasks.split(",")
#     train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
#     assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
#     train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
#     assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

#     train_datasets = []
#     for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
#         if task.lower() == "seqrec":
#             dataset = SeqRecFinetune(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

#         elif task.lower() == "item2index" or task.lower() == "index2item":
#             dataset = ItemFeatFinetune(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

#         elif task.lower() in ["inters2title", "inters2description", "intertitles2item"]:
#             dataset = FusionSeqRecFinetune(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

#         elif task.lower() in ["itemsearch", "query2item"]:
#             dataset = ItemSearchFinetune(args, task=task.lower(),mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

#         elif task.lower() == "preferenceobtain":
#             dataset = PreferenceObtainFinetune(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

#         else:
#             raise NotImplementedError
#         train_datasets.append(dataset)

#     train_data = ConcatDataset(train_datasets)

#     valid_data = SeqRecDataset(args,"valid",args.valid_prompt_sample_num)

#     return train_data, valid_data

def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--base_model", type=str,
                        default="../llama-7b/",
                        help="basic model path")
    parser.add_argument("--output_dir", type=str,
                        default="./ckpt/",
                        help="The output directory")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="",
                        help="data directory")
    parser.add_argument("--tasks", type=str, 
                        default='seqrec,itemsearch,inters2title,inters2description,preferenceobtain,item2index,index2item,intertitles2item,query2item',
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--train_data_sample_num", type=str, default="0,0,0,0,0,0,0,0,0",
                        help="the number of sampling data for each task")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="the item indices file")
    parser.add_argument("--user_index_file", type=str, default=".user-index.json", help="user indices file")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="dataloader prefetch_factor")

    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1,1,1,1,1,1,1,1,1",
                        help="the number of sampling prompts for each task")

    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")

    return parser

def parse_train_args(parser):
    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj", help="separate by comma")
    parser.add_argument("--lora_modules_to_save", type=str,
                        default="embed_tokens,lm_head", help="separate by comma")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--remove_unused_columns", action="store_true", default=False, help='if remove unused columns')

    return parser

def parse_rqvae_args(parser):
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    # parser.add_argument("--data_path", type=str,
    #                     default="../data/Games/Games.emb-llama-td.npy",
    #                     help="Input data path.")

    # parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=False, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:1", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_path", type=str, default="", help="output directory for model")
    parser.add_argument("--warmup", type=int, default=5, help="epochs for warmup")
    parser.add_argument("--item_model", type=str, default="", help="")
    parser.add_argument("--user_model", type=str, default="", help="")

    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="",
                        help="The checkpoint path")
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")


    return parser

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "item2index" or task.lower() == "index2item":
            dataset = ItemFeatDataset(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() in ["inters2title", "inters2description", "intertitles2item"]:
            dataset = FusionSeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() in ["itemsearch", "query2item"]:
            dataset = ItemSearchDataset(args, task=task.lower(),mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "preferenceobtain":
            dataset = PreferenceObtainDataset(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == 'usersearch':
            dataset = UserSearchDataset(args, prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() in ["pref2user", "user2pref"]:
            dataset = UserFeatDataset(args, task = task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args,"valid",args.valid_prompt_sample_num)

    return train_data, valid_data

def load_test_dataset(args):

    if args.test_task.lower() == "seqrec":
        test_data = SeqRecFinetune(args, mode="test", sample_num=args.sample_num)
    elif args.test_task.lower() == "itemsearch":
        test_data = ItemSearchDataset(args, mode="test", sample_num=args.sample_num)
    elif args.test_task.lower() == "fusionseqrec":
        test_data = FusionSeqRecDataset(args, mode="test", sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data

# def load_test_dataset(args):

#     if args.test_task.lower() == "seqrec":
#         test_data = SeqRecDataset(args, mode="test", sample_num=args.sample_num)
#     elif args.test_task.lower() == "itemsearch":
#         test_data = ItemSearchDataset(args, mode="test", sample_num=args.sample_num)
#     elif args.test_task.lower() == "fusionseqrec":
#         test_data = FusionSeqRecDataset(args, mode="test", sample_num=args.sample_num)
#     else:
#         raise NotImplementedError

#     return test_data

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
