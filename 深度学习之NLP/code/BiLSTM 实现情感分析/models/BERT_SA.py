import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


class Config:
    def __init__(self):
        # 训练配置
        self.seed = 22
        self.batch_size = 64
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.num_epochs = 100
        self.early_stop = 512
        self.max_seq_length = 128
        self.save_path = '../model_parameters/BERT_SA.bin'

        # 模型配置
        self.bert_hidden_size = 768
        self.model_path = 'bert-base-uncased'
        self.num_outputs = 2


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        tokenizer_class, bert_class, model_path = BertTokenizer, BertForSequenceClassification, config.model_path
        bert_config = BertConfig.from_pretrained(model_path, num_labels=config.num_outputs)
        self.tokenizer = tokenizer_class.from_pretrained(model_path)
        self.bert = bert_class.from_pretrained(model_path, config=bert_config).to(device)

    def forward(self, inputs):
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  add_special_tokens=True,
                                                  max_length=self.config.max_seq_length,
                                                  padding='max_length',
                                                  truncation='longest_first')

        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        att_mask = torch.tensor(tokens['attention_mask']).to(self.device)

        logits = self.bert(input_ids, attention_mask=att_mask).logits

        return logits