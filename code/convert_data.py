from transformers import LlamaTokenizer
import pickle
from tqdm import tqdm
from dataset import GPT2Dataset_onlyres
import argparse
from utils import get_multiround_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="ziqingyang/chinese-llama-lora-7b")
    parser.add_argument("--max_length",type=int,default=2048,help="max token length")
    parser.add_argument("--out_data_path",type=str,default='data/test/',help="the floader to load raw data and save preprocessed data")
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    datas = get_multiround_data(args.out_data_path + 'ift_data.pkl', 0)
    train_dataset = GPT2Dataset_onlyres(tokenizer, datas, args.max_length)

    pickle.dump(
        {
            "input_ids": train_dataset.input_ids,
            "labels": train_dataset.labels,
            "attention_mask": train_dataset.attention_mask
        },
        open(args.out_data_path + "llama_ift_data_ids.pkl", "wb")
    )

