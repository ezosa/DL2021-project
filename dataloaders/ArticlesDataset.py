from torch.utils.data import Dataset
import pandas as pd


class ArticlesDataset(Dataset):
    def __init__(self, ft_file, vocab):
        self.data = pd.read_feather(ft_file)
        self.vocab = vocab

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx, max_text_len=200):
        row = dict(self.data.iloc()[idx])
        combined_text = row['headline'] + ' ' + row['text']
        combined_text = combined_text.lower().split()
        combined_text = [self.vocab(w) if w in self.vocab else self.vocab['OOV'] for w in combined_text]
        if len(combined_text) > max_text_len:
                combined_text = combined_text[:max_text_len]
        else:
            combined_text.extend([self.vocab['OOV']]*(max_text_len-len(combined_text)))
        row['content'] = combined_text
        row['content_len'] = max_text_len
        return row


