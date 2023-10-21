import os
import json
import torch
import numpy as np
from collections import defaultdict
from transformers import BertTokenizer
from pytorch_GlobalPointer_triple_extraction.utils.train_utils import load_model_and_parallel
from pytorch_GlobalPointer_triple_extraction.model import GlobalPointerRe
from pytorch_GlobalPointer_Ner.utils.common_utils import read_json
from pytorch_GlobalPointer_Ner.globalpoint import GlobalPointerNer

class Dict2Class(dict):

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

def ner_predict(args, raw_text, model, tokenizer, id2tag, device):
  model.eval()
  with torch.no_grad():
      tokens = [i for i in raw_text]
      encode_dict = tokenizer.encode_plus(text=tokens,
                        max_length=args.max_seq_len,
                        padding='max_length',
                        truncation='longest_first',
                        is_pretokenized=True,
                        return_token_type_ids=True,
                        return_attention_mask=True)
      tokens = ['[CLS]'] + tokens + ['[SEP]']
      token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0)
      attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0)
      token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0)
      logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), None)
      batch_size = logits.size(0)
      pred_tmp = defaultdict(list)
      for i in range(batch_size):
        logit = logits[i, ...]
        for j in range(args.num_tags):
            logit_ = logit[j, :len(tokens), :len(tokens)]
            for start, end in zip(*np.where(logit_.cpu().numpy() > 0.5)):
                pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start-1])

      print(dict(pred_tmp))


def re_predict(args, 
      raw_text, 
      model, 
      tokenizer,
      id2tag,
      device):
  model.eval()
  with torch.no_grad():
      tokens = [i for i in raw_text]
      if len(tokens) > args.max_seq_len - 2:
        tokens = tokens[:args.max_seq_len - 2]
      tokens = ['[CLS]'] + tokens + ['[SEP]']
      token_ids = tokenizer.convert_tokens_to_ids(tokens)
      attention_masks = [1] * len(token_ids)
      token_type_ids = [0] * len(token_ids)
      if len(token_ids) < args.max_seq_len:
        token_ids = token_ids + [0] * (args.max_seq_len - len(tokens))
        attention_masks = attention_masks + [0] * (args.max_seq_len - len(tokens))
        token_type_ids = token_type_ids + [0] * (args.max_seq_len - len(tokens))
      assert len(token_ids) == args.max_seq_len
      assert len(attention_masks) == args.max_seq_len
      assert len(token_type_ids) == args.max_seq_len
      token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(device)
      attention_masks = torch.from_numpy(np.array(attention_masks, dtype=np.uint8)).unsqueeze(0).to(device)
      token_type_ids = torch.from_numpy(np.array(token_type_ids)).unsqueeze(0).to(device)
      entity_output, head_output, tail_output = model(token_ids, attention_masks, token_type_ids)

      cur_batch_size = entity_output.shape[0]
      spos = []
      subjects = []
      objects = []
      # print(entity_output.shape, head_output.shape, tail_output.shape)
      for i in range(cur_batch_size):
          example = raw_text
          l = len(example)
          subject = []
          object = []
          subject_ids = []
          object_ids = []
          spo = []
          single_entity_output = entity_output[i, ...]
          single_head_output = head_output[i, ...]
          single_tail_output = tail_output[i, ...]
          single_head_output = single_head_output[:, 1:l+1:, 1:l+1]
          single_tail_output = single_tail_output[:, 1:l+1:, 1:l+1]
          subject_entity_outpout = single_entity_output[:1, 1:l+1:, 1:l+1].squeeze()
          object_entity_output = single_entity_output[1:, 1:l+1:, 1:l+1].squeeze()
          # 注意这里阈值为什么是0
          subject_entity_outpout = np.where(subject_entity_outpout.cpu().numpy() > 0)
          object_entity_output = np.where(object_entity_output.cpu().numpy() > 0)
          for m,n in zip(*subject_entity_outpout):
            subject_ids.append((m, n))
          for m,n in zip(*object_entity_output):
            object_ids.append((m, n))
          for sh, st in subject_ids:
            for oh, ot in object_ids:
              # print(example[sh:st+1], example[oh:ot+1])
              # print(np.where(single_head_output[:, sh, oh].cpu().numpy() > 0))
              # print(np.where(single_tail_output[:, st, ot].cpu().numpy() > 0))
              subj = example[sh:st+1]
              obj = example[oh:ot+1]
              subject.append(subj)
              object.append(obj)

              re1 = np.where(single_head_output[:, sh, oh].cpu().numpy() > 0)[0]
              re2 = np.where(single_tail_output[:, st, ot].cpu().numpy() > 0)[0]
              res = set(re1) & set(re2)
              for r in res:
                spo.append((subj, id2tag[r], obj))

          subjects.append(subject)
          objects.append(object)
          spos.append(spo)

      print("文本：", raw_text)
      print('主体：', [list(set(i)) for i in subjects])
      print('客体：', [list(set(i)) for i in objects])
      print('关系：', spos)
      print("="*100)

def main():
  text = "冬十月，天子拜太祖兖州牧。十二月，雍丘溃，超自杀。夷邈三族。邈诣袁术请救，为其众所杀，兖州平，遂东略陈地。"

  bert_dir = "./model_hub/chinese-bert-wwm-ext/"

  # 实体识别相关信息
  # ======================================
  ner_path = "./pytorch_GlobalPointer_Ner/checkpoints/bert-1-eff/"    
  with open(os.path.join(ner_path, "args.json"), "r", encoding="utf-8") as fp:
    ner_args = json.load(fp)
  ner_args = Dict2Class(ner_args)
  ner_args.bert_dir = bert_dir
  tokenizer = BertTokenizer.from_pretrained(ner_args.bert_dir)

  ner_model = GlobalPointerNer(ner_args)
  ner_model_path = os.path.join(ner_path, "model.pt")
  ner_model, device = load_model_and_parallel(ner_model, ner_args.gpu_ids, ner_model_path)
  ner_args.data_dir = "./pytorch_GlobalPointer_Ner/data/guwen/"
  ner_data_path = os.path.join(ner_args.data_dir, 'mid_data')
  label_list = read_json(ner_data_path, 'labels')
  ner_tag2id = {}
  ner_id2tag = {}
  for k, v in enumerate(label_list):
      ner_tag2id[v] = k
      ner_id2tag[k] = v
  # ======================================

  # 关系抽取相关信息
  # ======================================
  re_path = "./pytorch_GlobalPointer_triple_extraction/checkpoints/bert/"
  with open(os.path.join(re_path, "args.json"), "r", encoding="utf-8") as fp:
      re_args = json.load(fp)
  re_args = Dict2Class(re_args)
  re_args.bert_dir = bert_dir
  re_model = GlobalPointerRe(re_args)
  re_model_path = os.path.join(re_path, "model.pt")
  re_model, device = load_model_and_parallel(re_model, re_args.gpu_ids, re_model_path)
  re_args.data_dir = "./pytorch_GlobalPointer_triple_extraction/data/guwen/"
  label_list = read_json(os.path.join(re_args.data_dir, 'mid_data'), 'predicates')
  re_tag2id = {}
  re_id2tag = {}
  for k,v in enumerate(label_list):
      re_tag2id[v] = k
      re_id2tag[k] = v
  # ======================================

  ner_predict(ner_args, text, ner_model, tokenizer, ner_id2tag, device)
  re_predict(re_args, text, re_model, tokenizer, re_id2tag, device)



if __name__ == "__main__":
  main()

