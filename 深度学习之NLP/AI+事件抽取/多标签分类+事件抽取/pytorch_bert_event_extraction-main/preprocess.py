import os
import json
import logging
from transformers import BertTokenizer
from collections import defaultdict
from utils import commonUtils
import config

logger = logging.getLogger(__name__)

class InputExample:
    def __init__(self, set_type, text, event_type=None, labels=None):
        self.set_type = set_type
        self.text = text
        self.event_type = event_type
        self.labels = labels


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class MRCBertFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 event_type=None,
                 start_ids=None,
                 end_ids=None):
        super(MRCBertFeature, self).__init__(token_ids=token_ids,
                                             attention_masks=attention_masks,
                                             token_type_ids=token_type_ids)
        self.event_type = event_type
        self.start_ids = start_ids
        self.end_ids = end_ids


class Processor:
  
    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.readlines()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        for raw_example in raw_examples:
          raw_example = eval(raw_example)
          text = raw_example['text']
          event_list = raw_example['event_list']
          for event in event_list:
            labels = []
            event_type = event['event_type']
            arguments = event['arguments']
            for argument in arguments:
              labels.append((event_type+'-'+argument['role'], argument['argument'], argument['argument_start_index']))
            examples.append(InputExample(set_type=set_type,
                            text=text,
                            event_type=event_type,
                            labels=labels))

        return examples


def convert_mrc_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                        max_seq_len, rolelabel2id, labels2query):
    set_type = example.set_type
    text_b = example.text
    event_type = example.event_type
    entities = example.labels
    # print(entities)
    features = []
    callback_info = []

    # 这里是text_b的tokens
    # tokens_b = commonUtils.fine_grade_tokenize(text_b, tokenizer)
    tokens_b = [i for i in text_b]
    assert len(tokens_b) == len(text_b)

    label_dict = defaultdict(list)

    # 这里的entities的格式是：实体类型 实体名 实体起始位置 实体结束位置
    for ent in entities:
        ent_type = ent[0]
        ent_start = ent[-1]
        ent_end = ent_start + len(ent[1]) - 1
        label_dict[ent_type].append((ent_start, ent_end, ent[1]))
    # print(label_dict)
    # 训练数据中构造
    # 每一类为一个 example
    # for _type in label_dict.keys():

    # 有多尔少个tokens，就有多少个标签，起始和结束位置都是
    start_ids = [0] * len(tokens_b)
    end_ids = [0] * len(tokens_b)

    # stop_mask_ranges = []

    # 这里加载的是每一类的问题
    # 比如 "DRUG": "找出药物：用于预防、治疗、诊断疾病并具有康复与保健作用的物质。"
    text_a = labels2query[event_type]
    # tokens_a = commonUtils.fine_grade_tokenize(text_a, tokenizer)
    tokens_a = [i for i in text_a]
    # 对于每一个类，将该实体在句子中的首尾置为1
    for _type in label_dict.keys():
      for _label in label_dict[_type]:
          start_ids[_label[0]] = rolelabel2id[_type]
          end_ids[_label[1]] = rolelabel2id[_type]

    # 输入的组成是：[CLS] text_a [SEP] text_b [SEP]，所以减去-3
    if len(start_ids) > max_seq_len - len(tokens_a) - 3:
        start_ids = start_ids[:max_seq_len - len(tokens_a) - 3]
        end_ids = end_ids[:max_seq_len - len(tokens_a) - 3]
        print('产生了不该有的截断')
    # 整合两个句子
    start_ids = [0] + [0] * len(tokens_a) + [0] + start_ids + [0]
    end_ids = [0] + [0] * len(tokens_a) + [0] + end_ids + [0]

    # pad
    # 整合之后进行padding
    if len(start_ids) < max_seq_len:
        pad_length = max_seq_len - len(start_ids)

        start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
        end_ids = end_ids + [0] * pad_length

    assert len(start_ids) == max_seq_len
    assert len(end_ids) == max_seq_len

    # 随机mask
    # if mask_prob:
    #     tokens_b = sent_mask(tokens_b, stop_mask_ranges, mask_prob=mask_prob)

    encode_dict = tokenizer.encode_plus(text=tokens_a,
                        text_pair=tokens_b,
                        max_length=max_seq_len,
                        padding="max_length",
                        truncation_strategy='only_second',
                        is_pretokenized=True,
                        return_token_type_ids=True,
                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
    if ex_idx < 3:
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text_all {tokens}')
        logger.info(f'text_b: {" ".join(tokens_b)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f'event type: {event_type}')
        logger.info(f"start_ids: {start_ids}")
        logger.info(f"end_ids: {end_ids}")

    # tmp_callback
    tmp_callback = (text_b, len(tokens_a) + 2, event_type)  # (text, text_offset, type, labels)
    tmp_callback_labels = []

    for _type in label_dict.keys():
      for _label in label_dict[_type]:
          tmp_callback_labels.append((_type, _label[0], _label[1]))
    # ('“众弈杯”2019年中国国际象棋甲级联赛武义专场开幕', 17, '组织行为-开幕', [('组织行为-开幕-活动名称', 0, 23)])
    tmp_callback += (tmp_callback_labels,)
    # print(tmp_callbsack)
    callback_info.append(tmp_callback)

    feature = MRCBertFeature(token_ids=token_ids,
                  attention_masks=attention_masks,
                  token_type_ids=token_type_ids,
                  event_type=label2id[event_type],
                  start_ids=start_ids,
                  end_ids=end_ids,
                  )

    features.append(feature)

    return features, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, rolelabel2id, labels2query):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []

    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_mrc_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            rolelabel2id=rolelabel2id, # 这个是实体类对应的id
            labels2query=labels2query, # 这个是问题类对应的id    
        )

        if feature is None:
            continue

        features.extend(feature)
        callback_info.extend(tmp_callback)

    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out


    out += (callback_info,)

    return out

def get_data(processor, raw_data_path, json_file, mode, rolelabel2id, labels2query, args):
    raw_examples = processor.read_json(os.path.join(raw_data_path, json_file))
    examples = processor.get_examples(raw_examples, mode)
    data = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, rolelabel2id, labels2query)
    save_path = os.path.join(args.data_dir, 'final_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    commonUtils.save_pkl(save_path, data, mode)
    return data


def save_file(filename, data):
    file = open(filename, 'w', encoding='utf-8')
    features, callback_info = data

    for feature, tmp_callback in zip(features, callback_info):
        text, offset, ent_type, gt_entities = tmp_callback
        start = feature.start_ids[offset:offset + len(text)]
        end = feature.end_ids[offset:offset + len(text)]
        file.write(ent_type + "\n")
        file.write("\t".join([str(i) for i in start]) + "\n")
        file.write("\t".join([str(i) for i in end]) + "\n")
        file.write("\t".join([word for word in text]) + "\n")
    file.close()

if __name__ == '__main__':
    args = config.Args().get_parser()
    commonUtils.set_logger(os.path.join(args.log_dir, 'preprocess.log'))

    args.data_dir = './data/'
    final_data_path = args.data_dir + 'final_data/'
    raw_data_path = args.data_dir + 'raw_data/'
    args.max_seq_len = 320
    
    label2id = {}
    id2label = {}
    with open(final_data_path+'labels.txt','r') as fp:
      labels = fp.read().strip().split('\n')
    for i,j in enumerate(labels):
      label2id[j] = i
      id2label[i] = j

    rolelabel2id = {}
    id2rolelabel = {}
    with open(final_data_path+'rolelabels.txt','r') as fp:
      rolelabels = fp.read().strip().split('\n')
    # 将0留出来
    for i,j in enumerate(rolelabels):
      rolelabel2id[j] = i+1
      id2rolelabel[i+1] = j
    role_labels = rolelabel2id.keys()
    print(final_data_path+'labels2query.json')
    with open(final_data_path+'labels2query.json','r') as fp:
      data = fp.read()
      labels2query = eval(data)
    with open(final_data_path+'labels2rolelabels.json','r') as fp:
      labels2rolelabels = json.loads(fp.read())

    # 这里简单设定问题的最大长度
    # query_max_len = max([len(x) for x in labels2query.values()])
    processor = Processor()

    train_data = get_data(processor, raw_data_path, "train.json", "train", rolelabel2id, labels2query, args)
    dev_data = get_data(processor, raw_data_path, "dev.json", "dev", rolelabel2id, labels2query, args)
    test_data = get_data(processor, raw_data_path, "dev.json", "test", rolelabel2id, labels2query, args)