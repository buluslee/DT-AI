import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.common_utils import sequence_padding


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, tokenizer=None, max_len=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path, tokenizer, max_len)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path, tokenizer, max_len):
        return file_path



# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename, tokenizer, max_len):
        data = []
        all_tokens = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            f = json.loads(f)
            for d in f:
              text = d['text']
              labels = d['labels']
              tokens = [i for i in text]
              if len(tokens) > max_len - 2:
                tokens = tokens[:max_len]
              tokens = ['[CLS]'] + tokens + ['[SEP]']
              all_tokens.append(tokens)
              token_ids = tokenizer.convert_tokens_to_ids(tokens)
              label = []
              for lab in labels:  # 这里需要加上CLS的位置, lab[3]不用加1，因为是实体结尾的后一位
                label.append([lab[2]+1, lab[3], lab[1]])
              data.append((token_ids, label))  # label为[[start, end, entity], ...]
        return data, all_tokens

class Collate:
  def __init__(self, max_len, tag2id, device):
      self.maxlen = max_len
      self.tag2id = tag2id
      self.device = device

  def collate_fn(self, batch):
      batch_labels = []
      batch_token_ids = []
      batch_attention_mask = []
      batch_token_type_ids = []
      for i, (token_ids, text_labels) in enumerate(batch):
          labels = np.zeros((len(self.tag2id), self.maxlen, self.maxlen))
          batch_token_ids.append(token_ids)  # 前面已经限制了长度
          batch_attention_mask.append([1] * len(token_ids))
          batch_token_type_ids.append([0] * len(token_ids))
          for start, end, label in text_labels:
            # 排除SEP及之后的
            if end >= self.maxlen - 1:
              continue
            label_id = self.tag2id[label]
            labels[label_id, start, end] = 1
          batch_labels.append(labels)
      batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=self.maxlen), dtype=torch.long, device=self.device)
      token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
      return batch_token_ids, attention_mask, token_type_ids, batch_labels


if __name__ == "__main__":
  from transformers import BertTokenizer
  max_len = 150
  tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext/vocab.txt')
  train_dataset, train_callback = MyDataset(file_path='data/cner/mid_data/train.json', 
              tokenizer=tokenizer, 
              max_len=max_len)
  print(train_dataset[0])

  with open('data/cner/mid_data/labels.json') as fp:
    labels = json.load(fp)
  id2tag = {}
  tag2id = {}
  for i,label in enumerate(labels):
    id2tag[i] = label
    tag2id[label] = i
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  collate = Collate(max_len=max_len, tag2id=tag2id, device=device)
  batch_size = 2
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 

  for i, batch in enumerate(train_dataloader):
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)
    print(batch[3].shape)
    break
