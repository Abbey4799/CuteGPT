from typing import List, Optional, Tuple

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import pickle
from tqdm import tqdm
import random

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from config import input_template_pool, template_pool, meta_prompt

def flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel
    
    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    # offset = 0
    if past_key_value is not None:
        offset = past_key_value[0].shape[-2]
        kv_seq_len += offset
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states,
                                                    key_states,
                                                    cos,
                                                    sin,
                                                    position_ids=position_ids)
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"
    assert past_key_value is None, "past_key_value is not supported"

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2) # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3) # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask


    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                device=qkv.device)
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                    indices, bsz, q_len),
                        'b s (h d) -> b s h d', h=nheads)
    return self.o_proj(rearrange(output,
                                    'b s h d -> b s (h d)')), None, None


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def flash_attn_prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                    inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask

def get_multiround_data(saved_dir, rank, max_training_samples = 1000):
    """
    This function retrieves multi-round conversation data from a saved directory.
    The resulting dialogues are stored in a list.

    Parameters:
    saved_dir (str): The directory path where the data is saved.
    rank (int): The rank or identifier of the process. Used for printing progress information.
    max_training_samples (int, optional): The maximum number of training samples to retrieve. Default is 1000. Set to -1 to retrieve all samples.

    Returns:
    datas (list): A list of multi-round dialogues.
    """

    print(saved_dir)
    with open(saved_dir, "rb") as f:
        res = pickle.load(f)
    random.shuffle(res)
    
    if max_training_samples != -1:
        res = res[:max_training_samples]

    if rank == 0:
        print('get_multiround_data...')
        print(len(res))
        pbar = tqdm(total=len(res), mininterval=0)

    begin_idx = 0
    datas = []
    while begin_idx < len(res):
        num = random.randint(1,10)
        samples = [element for lis in res[begin_idx: min(begin_idx + num, len(res))] for element in lis]
        prompts = [meta_prompt]
        for idx in range(len(samples)):
            if samples[idx]['input'] == '': 
                text = (random.choice(template_pool["wround_woinput"]).format(samples[idx]['instruction'], ''), samples[idx]['output'] + '<end>')
            else:
                text = (random.choice(template_pool["wround_winput"]).format(samples[idx]['instruction'], random.choice(input_template_pool).format(samples[idx]['input']), ''), samples[idx]['output'] + '<end>')
            prompts.append(text)

        begin_idx += num
        datas.append(prompts)

        if rank == 0:
            pbar.update(num)

    if rank == 0:
        print(datas[-1])
        print(len(datas))

    return datas



