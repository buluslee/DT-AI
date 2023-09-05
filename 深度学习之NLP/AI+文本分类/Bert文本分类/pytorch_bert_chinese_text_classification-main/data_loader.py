# coding=utf-8
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, tokenizer=None, max_len=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path


# 加载数据集
class CNEWSDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        data = []
        with open(filename, encoding='utf-8') as f:
            raw_data = f.readlines()
            for d in raw_data:
                d = d.strip().split('\t')
                text = d[1]
                label = d[0]
                data.append((text, label))
        return data


class CPWSDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        data = []
        with open(filename, encoding='utf-8') as f:
            raw_data = f.readlines()
            for d in raw_data:
                d = d.strip()
                d = d.split("\t")
                if len(d) == 2:
                    data.append((d[1], d[0]))
        return data




class Collate:
    def __init__(self, tokenizer, max_len, tag2id):
        self.tokenizer = tokenizer
        self.maxlen = max_len
        self.tag2id = tag2id

    def collate_fn(self, batch):
        batch_labels = []
        batch_token_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        for i, (text, label) in enumerate(batch):
            output = self.tokenizer.encode_plus(
                text=text,
                max_length=self.maxlen,
                padding="max_length",
                truncation='longest_first',
                return_token_type_ids=True,
                return_attention_mask=True
            )
            token_ids = output["input_ids"]
            token_type_ids = output["token_type_ids"]
            attention_mask = output["attention_mask"]
            batch_token_ids.append(token_ids)  # 前面已经限制了长度
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(self.tag2id[label])
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
        attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_data = {
            "token_ids":batch_token_ids,
            "attention_masks":attention_mask,
            "token_type_ids":token_type_ids,
            "labels":batch_labels
        }
        return batch_data


if __name__ == "__main__":
    from transformers import BertTokenizer

    max_len = 512
    tokenizer = BertTokenizer.from_pretrained('../model_hub/chinese-bert-wwm-ext')
    train_dataset = CNEWSDataset(file_path='data/cnews/cnews.train.txt')
    print(train_dataset[0])

    with open('data/cnews/labels.txt', 'r', encoding="utf-8") as fp:
        labels = fp.read().strip().split("\n")
    id2tag = {}
    tag2id = {}
    for i, label in enumerate(labels):
        id2tag[i] = label
        tag2id[label] = i
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate = Collate(tokenizer=tokenizer, max_len=max_len, tag2id=tag2id, device=device)
    batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

    for i, batch in enumerate(train_dataloader):
        print(batch["token_ids"].shape)
        print(batch["attention_masks"].shape)
        print(batch["token_type_ids"].shape)
        print(batch["labels"].shape)
        break
