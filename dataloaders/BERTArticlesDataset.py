import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class BERTArticlesDataset(Dataset):
    def __init__(self, df, label2id, tokenizer, max_len):
        self.data = df
        self.docs = list(self.data.text)
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_len = len(label2id)
        targets = list(self.data.codes)
        targets = [code.split() for code in targets]
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        doc = self.docs[idx]
        target = self.targets[idx]
        codes2id = np.array([self.label2id[code] for code in target if code in self.label2id])
        target = np.zeros(self.labels_len)
        target[codes2id] = 1

        #print("doc:", doc)
        encoding = self.tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        #print("attention_mask:", encoding['attention_mask'].shape)
        return {
            'doc_text': doc,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'targets': torch.tensor(target)
        }

