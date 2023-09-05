import sys

sys.path.append(r"./")
"""
该文件使用的data_loader.py里面的数据加载方式。
"""
# coding=utf-8
import json
import random
from pprint import pprint
import os
import logging
import shutil
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

import bert_config
import models
from utils import utils
from data_loader import CNEWSDataset, Collate, CPWSDataset

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader, device, model, optimizer):
        self.args = args
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def load_model(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    """
    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        tmp_checkpoint_path = checkpoint_path
        torch.save(state, tmp_checkpoint_path)
        if is_best:
            tmp_best_model_path = best_model_path
            shutil.copyfile(tmp_checkpoint_path, tmp_best_model_path)
    """

    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 100
        best_dev_micro_f1 = 0.0
        for epoch in range(args.train_epochs):
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(train_outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, micro_f1, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                    logger.info(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy,
                                                                                                   micro_f1, macro_f1))
                    if macro_f1 > best_dev_micro_f1:
                        logger.info("------------>保存当前最好的模型")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_micro_f1 = macro_f1
                        save_path = os.path.join(self.args.output_dir, args.data_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        checkpoint_path = os.path.join(save_path, 'best.pt')
                        print(checkpoint_path)
                        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, model):
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args, model):
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.max_seq_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(self.device)
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten().tolist()
            if len(outputs) != 0:
                outputs = [id2label[i] for i in outputs]
                return outputs
            else:
                return '不好意思，我没有识别出来'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels)
        return report


datasets = {
    "cnews": CNEWSDataset,
    "cpws": CPWSDataset
}

train_files = {
    "cnews": "cnews.train.txt",
    "cpws": "train_data.txt"
}

test_files = {
    "cnews": "cnews.test.txt",
    "cpws": "test_data.txt"
}


def main(args, tokenizer, device):
    dataset = datasets.get(args.data_name, None)
    train_file, test_file = train_files.get(args.data_name, None), test_files.get(args.data_name, None)
    if dataset is None:
        raise Exception("请输入正确的数据集名称")
    label2id = {}
    id2label = {}
    with open('./data/{}/labels.txt'.format(args.data_name), 'r', encoding="utf-8") as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    print(label2id)

    collate = Collate(tokenizer=tokenizer, max_len=args.max_seq_len, tag2id=label2id)

    train_dataset = dataset(file_path='data/{}/{}'.format(args.data_name, train_file))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=collate.collate_fn)
    test_dataset = dataset(file_path='data/{}/{}'.format(args.data_name, test_file))
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=collate.collate_fn)

    model = models.BertForSequenceClassification(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    if args.retrain:
        checkpoint_path = './checkpoints/{}/best.pt'.format(args.data_name)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info("加载模型继续训练，epoch:{} loss:{}".format(epoch, loss))

    trainer = Trainer(args, train_loader, test_loader, test_loader, device, model, optimizer)

    if args.do_train:
        # 训练和验证
        trainer.train()

    # 测试
    if args.do_test:
        logger.info('========进行测试========')
        checkpoint_path = './checkpoints/{}/best.pt'.format(args.data_name)
        model = trainer.load_model(model, checkpoint_path)
        total_loss, test_outputs, test_targets = trainer.test(model)
        accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
        logger.info(
            "【test】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(total_loss, accuracy, micro_f1,
                                                                                        macro_f1))
        report = trainer.get_classification_report(test_outputs, test_targets, labels)
        logger.info(report)

    # 预测
    if args.do_predict:
        checkpoint_path = './checkpoints/{}/best.pt'.format(args.data_name)
        model = trainer.load_model(model, checkpoint_path)
        line = test_dataset[0]
        text = line[0]
        print(text)
        result = trainer.predict(tokenizer, text, id2label, args, model)
        print("预测标签：", result[0])
        print("真实标签：", line[1])
        print("==========================")


if __name__ == '__main__':
    args = bert_config.Args().get_parser()
    utils.set_seed(args.seed)
    utils.set_logger(os.path.join(args.log_dir, 'main.log'))

    # processor = preprocess.Processor()

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    gpu_ids = args.gpu_ids.split(',')
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    main(args, tokenizer, device)
