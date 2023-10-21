from pprint import pprint
import os
import logging
import json
import shutil
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer

import bert_config
import preprocess
import dataset
import models
import utils

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = models.BertForRelationExtraction(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
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
                ids = train_data['ids'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids, ids)
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
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy, micro_f1, macro_f1))
                    if macro_f1 > best_dev_micro_f1:
                        logger.info("------------>保存当前最好的模型")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_micro_f1 = macro_f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
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
                ids = dev_data['ids'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids, ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(),axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint_path)
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
                ids = test_data['ids'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids, ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(),axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args, ids):
        model = self.model
        optimizer = self.optimizer
        checkpoint = os.path.join(args.output_dir, 'best.pt')
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint)
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
            token_ids = inputs['input_ids'].to(self.device).long()
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            ids = torch.from_numpy(np.array([[x+1 for x in ids]])).to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids, ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(),axis=1).flatten().tolist()
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


if __name__ == '__main__':
    args = bert_config.Args().get_parser()
    utils.utils.set_seed(args.seed)
    utils.utils.set_logger(os.path.join(args.log_dir, 'main.log'))

    processor = preprocess.Processor()

    label2id = {}
    id2label = {}
    with open('./data/rel_dict.json', 'r') as fp:
        labels = json.loads(fp.read())
    for k, v in labels.items():
        label2id[k] = v
        id2label[v] = k
    print(label2id)

    # train_out = preprocess.get_out(processor, './data/train.txt', args, id2label, 'train')
    # dev_out = preprocess.get_out(processor, './data/test.txt', args, id2label, 'dev')
    # test_out = preprocess.get_out(processor, './data/test.txt', args, id2label, 'test')
    #
    # train_features, train_callback_info = train_out
    # train_dataset = dataset.ReDataset(train_features)
    # train_sampler = RandomSampler(train_dataset)
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=args.train_batch_size,
    #                           sampler=train_sampler,
    #                           num_workers=2)
    #
    # dev_features, dev_callback_info = dev_out
    # dev_dataset = dataset.ReDataset(dev_features)
    # dev_loader = DataLoader(dataset=dev_dataset,
    #                         batch_size=args.eval_batch_size,
    #                         num_workers=2)
    #
    # test_features, test_callback_info = dev_out
    # test_dataset = dataset.ReDataset(test_features)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=args.eval_batch_size,
    #                          num_workers=2)
    #
    # trainer = Trainer(args, train_loader, dev_loader, test_loader)
    # # 训练和验证
    # trainer.train()
    #
    # # 测试
    # logger.info('========进行测试========')
    # checkpoint_path = './checkpoints/best.pt'
    # total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    # accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
    # logger.info(
    #     "【test】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(total_loss, accuracy, micro_f1, macro_f1))
    # report = trainer.get_classification_report(test_outputs, test_targets, labels)
    # logger.info(report)

    # 预测
    trainer = Trainer(args, None, None, None)
    checkpoint_path = './checkpoints/best.pt'
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    with open(os.path.join('./data/predict.txt'), 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split('\t')
            label = line[0]
            text = line[1]
            ids = [int(line[2]),int(line[3]),int(line[4]),int(line[5])]
            logger.info(text)
            result = trainer.predict(tokenizer, text, id2label, args, ids)
            logger.info("预测标签：" + "".join(result))
            logger.info("真实标签：" + id2label[int(line[0])])
            logger.info("==========================")

    # 预测单条
    # text = '1	1537年，#亨利八世#和他的第三个王后$简·西摩$生了一个男孩：爱德华（后来的爱德华六世）。	6	11	20	25'
    text = '1537年，#亨利八世#和他的第三个王后$简·西摩$生了一个男孩：爱德华（后来的爱德华六世）。'
    ids = [6, 11, 20, 25]
    print('预测标签：', trainer.predict(tokenizer, text, id2label, args, ids))
    print('真实标签：', id2label[1])
