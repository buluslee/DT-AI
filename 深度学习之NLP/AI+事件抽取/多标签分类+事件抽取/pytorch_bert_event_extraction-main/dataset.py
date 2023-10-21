import torch
from torch.utils.data import Dataset,DataLoader
# 这里要显示的引入BertFeature，不然会报错
from preprocess import MRCBertFeature
from utils import commonUtils
import pickle


class MrcDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.start_ids = [torch.tensor(example.start_ids).long() for example in features]
        self.end_ids = [torch.tensor(example.end_ids).long() for example in features]


    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}


        data['start_ids'] = self.start_ids[index]
        data['end_ids'] = self.end_ids[index]

        return data

if __name__ == '__main__':
  with open('./data/final_data/dev.pkl','rb') as fp:
    train_data = pickle.load(fp)
  train_features, train_callback_info = train_data
  for train_feature in train_features:
    print(train_feature.token_ids)
    print(train_feature.attention_masks)
    print(train_feature.token_type_ids)
    print(train_feature.start_ids)
    print(train_feature.end_ids)
    break
  for tmp_callback_info in train_callback_info:
    print(tmp_callback_info)
    text, offset, event_type, entities = tmp_callback_info
    print(text)
    print(offset)
    print(event_type)
    print(entities)
    break

  mrcDataset = MrcDataset(train_features)
  print(len(mrcDataset))
  # print(mrcDataset[0])
  