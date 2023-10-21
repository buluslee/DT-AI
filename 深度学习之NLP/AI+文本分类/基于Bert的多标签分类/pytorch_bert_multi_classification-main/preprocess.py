import sys
sys.path.append('..')
import os
import logging
from transformers import BertTokenizer
import bert_config
import numpy as np
from utils import utils

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels


class Processor:

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.read().strip()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        # 这里是从json数据中的字典中获取
        for line in raw_examples.split('\n'):
            line = eval(line)
            labels = []
            if len(line['event_list']) != 0:
                for tmp in line['event_list']:
                    labels.append(tmp['event_type'])
            examples.append(InputExample(set_type=set_type,
                                         text=line['text'],
                                         labels=labels))
        return examples


def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer, max_seq_len, label2id):
    set_type = example.set_type
    raw_text = example.text
    labels = example.labels
    # 文本元组
    callback_info = (raw_text,)
    callback_labels = labels
    callback_info += (callback_labels,)
    # 转换为one-hot编码
    label_ids = [0 for _ in range(len(label2id))]
    for label in labels:
        label_ids[label2id[label]] = 1

    encode_dict = tokenizer.encode_plus(text=raw_text,
                                        add_special_tokens=True,
                                        max_length=max_seq_len,
                                        truncation_strategy='longest_first',
                                        padding="max_length",
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    while len(token_ids) < max_seq_len:
        token_ids.append(0)
        attention_masks.append(0)
        token_type_ids.append(0)

    assert len(token_ids) == max_seq_len
    assert len(attention_masks) == max_seq_len
    assert len(token_type_ids) == max_seq_len
    
    if ex_idx < 3:
        decode_text = tokenizer.decode(np.array(token_ids)[np.where(np.array(attention_masks) == 1)[0]].tolist())
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f"text: {decode_text}")
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"labels: {label_ids}")

    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids,
    )

    return feature, callback_info

def convert_examples_to_features(examples, max_seq_len, bert_dir, label2id):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            label2id = label2id,
        )
        if feature is None:
            continue

        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def get_out(processor, json_path, args, label2id, mode):
    raw_examples = processor.read_json(json_path)

    examples = processor.get_examples(raw_examples, mode)
    for i, example in enumerate(examples):
        print(example.text)
        print(example.labels)
        if i == 5:
            break
    out = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, label2id)
    return out


if __name__ == '__main__':
    args = bert_config.Args().get_parser()
    args.log_dir = './logs/'
    args.max_seq_len = 128
    args.bert_dir = 'models'
    utils.set_logger(os.path.join(args.log_dir, 'preprocess.log'))
    logger.info(vars(args))

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('./data/final_data/labels.txt','r') as fp:
        labels = fp.read().strip().split('\n')
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[id] = label
    print(label2id)

    train_out = get_out(processor, './data/raw_data/train.json', args, label2id, 'train')
    dev_out = get_out(processor, './data/raw_data/dev.json', args, label2id, 'dev')
