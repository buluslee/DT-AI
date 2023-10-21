import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.common_utils import sequence_padding


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, **kwargs):
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
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        examples = []
        with open(filename, encoding='utf-8') as f:
            raw_examples = f.readlines()
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            # print(i,item)
            item = json.loads(item)
            text = item['text']
            spo_list = item['spo_list']
            labels = [] # [subject, predicate, object]
            for spo in spo_list:
                subject = spo['subject']
                object = spo['object']
                predicate = spo['predicate']
                labels.append([subject, predicate, object])
            examples.append((text, labels))
        return examples

# 加载古文数据集
class GuwenDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        examples = []
        with open(filename, encoding='utf-8') as f:
            raw_examples = f.readlines()
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            # print(i,item)
            item = json.loads(item)
            text = item['text']
            spo = item['spo_list']
            labels = [] # [subject, predicate, object]
            subject = spo['subject']
            object = spo['object']
            predicate = spo['predicate']
            labels.append([subject, predicate, object])
            examples.append((text, labels))
        return examples

class Collate:
  def __init__(self, max_len, tag2id, device, tokenizer):
      self.maxlen = max_len
      self.tag2id = tag2id
      self.id2tag = {v:k for k,v in tag2id.items()}
      self.device = device
      self.tokenizer = tokenizer

  def collate_fn(self, batch):
      def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
      batch_head_labels = []
      batch_tail_labels = []
      batch_entity_labels = []
      batch_token_ids = []
      batch_attention_mask = []
      batch_token_type_ids = []
      callback = []
      for i, (text, text_labels) in enumerate(batch):
          if len(text) > self.maxlen - 2:
            text = text[:self.maxlen - 2]
          tokens = [i for i in text]
          tokens = ['[CLS]'] + tokens + ['[SEP]']
          spoes = set()
          callback_text_labels = []
          for s, p, o in text_labels:
            p = self.tag2id[p]
            s = [i for i in s]
            o = [i for i in o]
            s_idx = search(s, tokens)  # 主体的头
            o_idx = search(o, tokens)  # 客体的头
            if s_idx != -1 and o_idx != -1:
              callback_text_labels.append(("".join(s), self.id2tag[p], "".join(o)))
              spoes.add((s_idx, s_idx + len(s) - 1, p, o_idx, o_idx + len(o) - 1))
          # print(text_labels)
          # print(text)
          # print(spoes)
          # 构建标签
          entity_labels = [set() for _ in range(2)]  # [主体， 客体]
          head_labels = [set() for _ in range(len(self.tag2id))] # 每个关系中主体和客体的头
          tail_labels = [set() for _ in range(len(self.tag2id))] # 每个关系中主体和客体的尾
          for sh, st, p, oh, ot in spoes:
              entity_labels[0].add((sh, st))
              entity_labels[1].add((oh, ot))
              head_labels[p].add((sh, oh))
              tail_labels[p].add((st, ot))
          
          
          for label in entity_labels + head_labels + tail_labels:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充
          
          # entity_labels:(2, 1, 2) head_labels:(49, 1, 2) tail_labels:(49, 1, 2)
          """
          对于entity_labels而言，第一个集合是主体，第二个集合是客体，使用pading补全到相同长度
          [{(0, 2)}, {(21, 22), (5, 9)}]
          [[[ 0  2]
            [ 0  0]]

          [[21 22]
            [ 5  9]]]
          [['九玄珠', '连载网站', '纵横中文网'], ['九玄珠', '作者', '龙马']]
          """

          entity_labels = sequence_padding([list(l) for l in entity_labels])  # [subject/object=2, 实体个数, 实体起终点]
          head_labels = sequence_padding([list(l) for l in head_labels])  # [关系个数, 该关系下subject/object配对数, subject/object起点]
          tail_labels = sequence_padding([list(l) for l in tail_labels])  # [关系个数, 该关系下subject/object配对数, subject/object终点]


          token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
          batch_token_ids.append(token_ids)  # 前面已经限制了长度
          batch_attention_mask.append([1] * len(token_ids))
          batch_token_type_ids.append([0] * len(token_ids))
          batch_head_labels.append(head_labels)
          batch_tail_labels.append(tail_labels)
          batch_entity_labels.append(entity_labels)
          callback.append((text, callback_text_labels))
      batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=self.maxlen), dtype=torch.long, device=self.device)
      token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=self.maxlen), dtype=torch.long, device=self.device)
      batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2), dtype=torch.float, device=self.device)
      batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2), dtype=torch.float, device=self.device)
      batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2), dtype=torch.float, device=self.device)

      return batch_token_ids, attention_mask, token_type_ids, batch_head_labels, batch_tail_labels, batch_entity_labels, callback


if __name__ == "__main__":
  from transformers import BertTokenizer
  max_len = 256
  tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext/vocab.txt')
  train_dataset = MyDataset(file_path='data/ske/raw_data/train_data.json', 
              tokenizer=tokenizer, 
              max_len=max_len)
  # print(train_dataset[0])

  with open('data/ske/mid_data/predicates.json') as fp:
    labels = json.load(fp)
  id2tag = {}
  tag2id = {}
  for i,label in enumerate(labels):
    id2tag[i] = label
    tag2id[label] = i
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  collate = Collate(max_len=max_len, tag2id=tag2id, device=device, tokenizer=tokenizer)
  # collate.collate_fn(train_dataset[:16])
  batch_size = 2
  train_dataset = train_dataset[:10]
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 

  """
  torch.Size([2, 256])
  torch.Size([2, 256])
  torch.Size([2, 256])
  torch.Size([2, 49, 1, 2])
  torch.Size([2, 49, 1, 2])
  torch.Size([2, 2, 1, 2])
  """
  for i, batch in enumerate(train_dataloader):
    leng = len(batch) - 1
    for j in range(leng):
      print(batch[j].shape)
    break