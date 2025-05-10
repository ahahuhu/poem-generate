import torch
from transformers import BertTokenizer
import numpy as np



class PoemDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        data = np.load(data_path)
        data = [str(poem) for poem in data]
        self.data = data # 为一个数组，数组里面的元素是一首诗的各个字符串

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        poem: str = self.data[index]
        return (index, poem)

    def collate_fn(self, batch):
        idx = [item[0] for item in batch]
        poem = [item[1] for item in batch]
        encoding = self.tokenizer(poem, return_tensors='pt', padding=True,truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sent_ids': idx
        }

        return batched_data
    