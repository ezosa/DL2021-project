from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ArticlesDataset(Dataset):
    def __init__(self, csv_file, vocab, label2id):
        self.data = pd.read_csv(csv_file)
        self.vocab = vocab
        self.label2id = label2id
        self.labels_len = len(label2id)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx, max_text_len=200):
        row = dict(self.data.iloc()[idx])
        combined_text = row['headline'] + ' ' + row['text']
        combined_text = combined_text.lower().split()
        combined_text = [self.vocab[w] if w in self.vocab else self.vocab['OOV'] for w in combined_text]
        if len(combined_text) > max_text_len:
                combined_text = combined_text[:max_text_len]
        else:
            combined_text.extend([self.vocab['OOV']]*(max_text_len-len(combined_text)))
        row['content'] = combined_text
        row['content_len'] = max_text_len
        codes = row['codes']
        codes = codes.split()
        codes2id = np.array([self.label2id[code] for code in codes if code in self.label2id])
        binary_label = np.zeros(self.labels_len)
        binary_label[codes2id] = 1
        row['binary_label'] = binary_label
        return row


