import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import copy


class GPT2Dataset_onlyres(Dataset):
    '''
    Dataset construction for training GPT-2 model, without padding. Truncation is done using the end-of-sequence (EOS) token, and only the loss for the response is computed.
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index = 0
        
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._preprocess()

    def _preprocess(self):
        self.input_ids = []

        self.labels = []
        meta_prompt = self.datas[0][0]
        meta_tokens = self.tokenizer(meta_prompt, padding=False, truncation=False, add_special_tokens=False)
        meta_tokens = meta_tokens["input_ids"][-self.max_length//3:]
        
        for data in tqdm(self.datas):
            sample_input_ids = copy.copy(meta_tokens)
            sample_labels = [-100] * len(sample_input_ids)

            for idx, item in enumerate(data):
                if idx > 0:
                    input, output = item[0], item[1]

                    input_tokens = self.tokenizer(input, padding=False, truncation=False, add_special_tokens=False)
                    input_tokens = input_tokens["input_ids"][:self.max_length//3]

                    len_input = len(input_tokens)
                    output_tokens = self.tokenizer(output, padding=False, truncation=False, add_special_tokens=False)
                    output_tokens = output_tokens["input_ids"][:2 * (self.max_length//3) - 1]

                    sample_input_ids += input_tokens + output_tokens
                    sample_labels += [-100] * len_input + output_tokens
            
            if len(sample_input_ids) != len(meta_tokens):
                self.input_ids += sample_input_ids
                self.labels += sample_labels
            
                self.input_ids += [self.tokenizer.eos_token_id]
                self.labels += [self.tokenizer.eos_token_id]

        self.attention_mask = [1] * len(self.input_ids)

    def __len__(self):
        return len(self.input_ids) // self.max_length

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index * self.max_length : (index + 1) * self.max_length]), \
                torch.tensor(self.labels[index * self.max_length : (index + 1) * self.max_length]), \
                    torch.tensor(self.attention_mask[index * self.max_length : (index + 1) * self.max_length])


class BertDataset_onlyres(Dataset):
    '''
    Padding is applied between each sample, and the length of each sample does not exceed max_length. Only the loss for the response is computed.
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print('BertDataset_onlyres finished..')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):

        meta_prompt = self.datas[0][0]
        meta_tokens = self.tokenizer(meta_prompt, padding=False, truncation=False, add_special_tokens=False)
        meta_tokens = meta_tokens["input_ids"][-self.max_length//3:]
        
        data = self.datas[index]
        sample_input_ids = copy.copy(meta_tokens)
        sample_labels = [-100] * len(sample_input_ids)

        for idx, item in enumerate(data):
            if idx > 0:
                input, output = item[0], item[1]
                input_tokens = self.tokenizer(input, padding=False, truncation=False, add_special_tokens=False)
                input_tokens = input_tokens["input_ids"][:self.max_length//3]

                len_input = len(input_tokens)
                output_tokens = self.tokenizer(output, padding=False, truncation=False, add_special_tokens=False)
                output_tokens = output_tokens["input_ids"][:2 * (self.max_length//3) - 1]

                sample_input_ids += input_tokens + output_tokens
                sample_labels += [-100] * len_input + output_tokens

        sample_input_ids += [self.tokenizer.eos_token_id]
        sample_labels += [self.tokenizer.eos_token_id]
        sample_attention_mask = [1] * len(sample_input_ids)
        
        sample_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids))
        sample_labels += [-100] * (self.max_length - len(sample_labels))
        sample_attention_mask += [0] * (self.max_length - len(sample_attention_mask))
        
        sample_input_ids = sample_input_ids[:self.max_length]
        sample_labels = sample_labels[:self.max_length]
        sample_attention_mask = sample_attention_mask[:self.max_length]


        return torch.tensor(sample_input_ids), torch.tensor(sample_labels), torch.tensor(sample_attention_mask)



class DatasetIds(Dataset):
    '''
    Dataset construction for training GPT-2 model, without padding. Truncation is done using the end-of-sequence (EOS) token.
    This dataset directly loads preprocessed data, eliminating the need for waiting.
    '''
    def __init__(self,  tokenizer, datas, max_length, **kwargs):
        super().__init__()
        self.input_ids = datas['input_ids']
        self.attention_mask = datas['attention_mask']
        self.labels = datas['labels']
        self.max_length = max_length

    def __len__(self):
        return len(self.input_ids) // self.max_length

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index * self.max_length : (index + 1) * self.max_length]), \
                torch.tensor(self.labels[index * self.max_length : (index + 1) * self.max_length]), \
                    torch.tensor(self.attention_mask[index * self.max_length : (index + 1) * self.max_length])
