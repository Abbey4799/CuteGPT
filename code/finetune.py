import os
import argparse
import sys
import torch
import deepspeed
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import pickle

from dataset import GPT2Dataset_onlyres, BertDataset_onlyres, DatasetIds
from utils import flash_attn_forward, flash_attn_prepare_decoder_attention_mask, get_multiround_data
from peft import (
    get_peft_model,
    PeftModel
)

import random
from config import lora_config, DS_CONFIG_lora, DS_CONFIG_ft


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = flash_attn_prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flash_attn_forward


def get_model_layers(model):
    layers = [["", model]]
    i = 0
    while i < len(layers):
        for nc, lc in layers[i][1].named_children():
            layers.append([f"{layers[i][0]}.{nc}" if layers[i][0] else nc, lc])
        i += 1
    return layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument("--seed",type=int,default=10,help="random seed")
    parser.add_argument("--max_epoches",type=int,default=5,help="max epoches to run dataloader")
    parser.add_argument("--max_training_samples",type=int,default=-1,help="max number of training samples")
    parser.add_argument("--data_path",type=str,default='',help="the floader to load training data")
    parser.add_argument("--model_path",type=str,default='',help="the floader to load model")
    parser.add_argument("--max_length",type=int,default=1024,help="max token length")
    parser.add_argument("--use_flash_attention",action="store_true",help="whether to use flash attention")
    parser.add_argument("--use_lora",action="store_true",help="Whether to use LoRa, the default is to perform full Finetune")
    parser.add_argument("--load_lora",action="store_true",help="whether load ckpts")
    parser.add_argument("--load_lora_path",type=str,default="",help="the floader to load lora ckpts(.pt)")
    parser.add_argument("--save_dir",type=str,default="ckp/",help="the floader to save ckpts(.pt)")
    parser.add_argument("--save_name",type=str,default="bloom_new",help="the floader extension name")
    parser.add_argument("--save_steps",type=int,default=1000,help="how many step to save a model")
    parser.add_argument("--dataset_type",choices=['GPT2Dataset_onlyres','BertDataset_onlyres', 'DatasetIds'],help="The type of dataset for dataloader")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.use_lora:
        DS_CONFIG = DS_CONFIG_lora
    else:
        DS_CONFIG = DS_CONFIG_ft
    print(DS_CONFIG)

    random.seed(args.seed)

    for path in [args.save_dir, args.save_dir + args.save_name]:
        if not os.path.exists(path):
            os.mkdir(path)

    device = torch.device("cuda")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()

    model_name = args.model_path
    print('model_name:',model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if not args.use_lora:
        st = ["<end>"]
        tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + st})
        print(tokenizer.additional_special_tokens)
    print('additional_special_tokens:', tokenizer.additional_special_tokens)

    if 'Ids' in args.dataset_type:
        # preprocessed data
        assert('llama_ift_data_ids.pkl' in args.data_path)
        datas = pickle.load(open(args.data_path, "rb"))
    else:
        assert('llama_ift_data_ids.pkl' not in args.data_path)
        datas = get_multiround_data(args.data_path,torch.distributed.get_rank(), max_training_samples=args.max_training_samples)

    train_dataset = eval(f"{args.dataset_type}")(
        tokenizer,
        datas, # your data preprocessing function
        args.max_length # your max input length
    )
    print('dataset loaded!')

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        sampler=train_sampler,
        batch_size=DS_CONFIG["train_micro_batch_size_per_gpu"]
    )

    model = LlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
    if args.use_flash_attention:
        print('using flash attn!!')
        replace_llama_attn_with_flash_attn()
    else:
        print('not using flash attn!!')

    if not args.use_lora:
        model.resize_token_embeddings(len(tokenizer))
        model_layers = get_model_layers(model)
        for layer in model_layers:
            if layer[0] in ['model.embed_tokens']:
                begin_idx = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens[0])
                end_idx = begin_idx + len(tokenizer.additional_special_tokens)
                print('normalize special token...')
                print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.tensor([begin_idx]))))
                # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.tensor([end_idx - 1]))))
                torch.nn.init.normal_(layer[1].weight.data[begin_idx:end_idx], std=1e-6)
    else:
        if args.load_lora:
            # load lora parameter
            print('parameter loaded!')
            print(args.load_lora_path)
            model = PeftModel.from_pretrained(model, args.load_lora_path, is_trainable= True)
        else:
            # training from scratch
            print('training from scratch')
            model = get_peft_model(model, lora_config)

    engine, _, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model, 
        model_parameters=model.parameters(),
    )
    print("model loaded.")

    args.max_steps = args.max_epoches * len(train_dataloader)

    global_step = 0
    engine.train()
    for epoch in range(args.max_epoches):
        losses = []
        if torch.distributed.get_rank() != -1:
            train_sampler.set_epoch(epoch)
        if torch.distributed.get_rank() == 0:
            pbar = tqdm(range(len(train_dataloader)))

        for batch in train_dataloader:
            loss = engine(
                input_ids = batch[0].to(device),
                labels = batch[1].to(device),
                attention_mask = batch[2].to(device),
                use_cache=False
            ).loss

            engine.backward(loss)
            engine.step()
            engine.zero_grad()

            global_step += 1
            losses.append(loss.item())
            if global_step % args.save_steps == 0:
                dist.barrier()
                if torch.distributed.get_rank() == 0:
                    if args.use_lora:
                        model.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_{global_step}")
                    else:
                        engine.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_{global_step}")
                    tokenizer.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_{global_step}")
                dist.barrier()

            if torch.distributed.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"loss: {sum(losses[-200: ]) / len(losses[-200: ])}")

            if global_step >= args.max_steps:
                break
        

        dist.barrier()
        if torch.distributed.get_rank() == 0:
            if args.use_lora:
                model.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_epoch{epoch}")
            else:
                engine.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_epoch{epoch}")
            tokenizer.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_epoch{epoch}")
        dist.barrier()

        if torch.distributed.get_rank() == 0:
            pbar.close()
        if global_step >= args.max_steps:
            break

