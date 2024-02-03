'''
Date: 2024-02-02 15:16:17
LastEditors: Dylancer1998 bodcoder@gmail.com
LastEditTime: 2024-02-02 17:22:06
'''

import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset
from megatron_patch.tokenizer import get_tokenizer


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, max_length: int=512, split: str='train'):
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        self.max_length = max_length
        
        list_data_dict = load_dataset(
            'json',
            data_files=path[0],
            split=split,
        )
        
        self.prompts = []
        for idx in range(len(list_data_dict['input'])):
            prompt = list_data_dict['instruction'][idx] + list_data_dict['input'][idx]
            if len(self.tokenizer.encode(prompt)) > self.max_length:
                continue
            self.prompts.append(prompt)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)