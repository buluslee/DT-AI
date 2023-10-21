import os
import json
import logging
from transformers import BertTokenizer
from utils import cut_sentence, common_utils

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels

    def __repr__(self):
        string = ""
        for key, value in self.__dict__.items():
            string += f"{key}: {value}\n"
        return f"<{string}>"


class NerProcessor:
    def __init__(self, cut_sent=True, cut_sent_len=256):
        self.cut_sent = cut_sent
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = json.load(f)
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            # print(i,item)
            text = item['text']
            if self.cut_sent:
                sentences = cut_sentence.cut_sent_for_bert(text, self.cut_sent_len)
                start_index = 0

                for sent in sentences:
                    labels = cut_sentence.refactor_labels(sent, item['labels'], start_index)
                    start_index += len(sent)
                    
                    examples.append(InputExample(set_type=set_type,
                                                 text=sent,
                                                 labels=labels))
            else:
                labels = item['labels']
                if len(labels) != 0:
                    labels = [(label[1],label[4],label[2]) for label in labels]
                examples.append(InputExample(set_type=set_type,
                                             text=text,
                                             labels=labels))

        return examples



if __name__ == "__main__":
  nerProcessor = NerProcessor(cut_sent=True, cut_sent_len=150)
  raw_examples = nerProcessor.read_json('data/people-daily/mid_data/train.json')
  examples = nerProcessor.get_examples(raw_examples, set_type="train")
  print(examples[0])
