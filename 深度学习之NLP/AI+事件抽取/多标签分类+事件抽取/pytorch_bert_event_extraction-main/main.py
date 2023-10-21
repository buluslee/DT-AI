# coding=utf-8
import os
import logging
import numpy as np
import torch
import json
import random
from pprint import pprint

from utils import commonUtils, trainUtils
import config
import dataset
import bertMrc
from preprocess import MRCBertFeature
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from seqeval.metrics.sequence_labeling import get_entities

args = config.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)
commonUtils.set_logger(os.path.join(args.log_dir, 'bertMrc.log'))


def convert_value_to_bio(start, end, text, id2rolelabel):
    """
    text = '我爱北京的烤鸭'
    labels = {1:'地点',2:'食品'}
    start = [0,0,1,0,0,2,0]
    end = [0,0,0,1,0,0,2]
    convert_value_to_bio(start,end,text,labels)
    ['0', '0', 'B-地点', 'I-地点', '0', 'B-食品', 'I-食品']
    """
    res = ['O'] * len(start)
    length = len(start)
    for i in range(length):
        if start[i] != 0:
            for j in range(i, length):
                if start[i] == end[j]:
                    label = id2rolelabel[start[i]]
                    label = label.replace('-', '_')
                    res[i] = 'B-' + label
                    for k in range(i + 1, j + 1):
                        res[k] = 'I-' + label
                    break
    return res


class BertForMrc:
    def __init__(self, model, train_loader, dev_loader, test_loader, args):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model, self.device = trainUtils.load_model_and_parallel(self.model, args.gpu_ids)
        if len(self.train_loader) != 0:
            self.t_total = len(self.train_loader) * self.args.train_epochs
        self.optimizer, self.scheduler = trainUtils.build_optimizer_and_scheduler(self.args, self.model, self.t_total)

    def train(self, model_name):
        global_step = 0
        eval_steps = self.t_total // 3
        logger.info('每{}个step进行验证。。。'.format(eval_steps))
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                start_logits, end_logits = self.model(batch_data['token_ids'], batch_data['attention_masks'],
                                                      batch_data['token_type_ids'], batch_data['start_ids'],
                                                      batch_data['end_ids'])
                loss = self.model.loss(batch_data['start_ids'], batch_data['end_ids'], start_logits, end_logits,
                                       batch_data['token_type_ids'])
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                # loss.backward(loss.clone().detach())
                # print(loss.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('[train] epoch:{}/{} step:{}/{} loss:{:.6f}'.format(epoch, self.args.train_epochs,
                                                                            global_step, self.t_total, loss.item()))
                global_step += 1
                if global_step % eval_steps == 0:
                    dev_loss, accuracy, precision, recall, f1 = self.dev()
                    logger.info('[dev] loss:{:.6f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
                        dev_loss, accuracy, precision, recall, f1))
                    if f1 > best_f1:
                        best_f1 = f1
                        trainUtils.save_model(self.args, self.model, model_name, global_step)

    def dev(self):
        s_logits, e_logits = None, None
        true_s_logits, true_e_logits = None, None
        self.model.eval()
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                batch_size, max_seq_length = dev_batch_data['token_ids'].size()
                start_logits, end_logits = self.model(dev_batch_data['token_ids'],
                                                      dev_batch_data['attention_masks'],
                                                      dev_batch_data['token_type_ids'],
                                                      dev_batch_data['start_ids'],
                                                      dev_batch_data['end_ids'])
                loss = self.model.loss(dev_batch_data['start_ids'], dev_batch_data['end_ids'], start_logits, end_logits,
                                       dev_batch_data['token_type_ids'])
                start_logits = start_logits.reshape(batch_size, max_seq_length, -1).detach().cpu().numpy()
                end_logits = end_logits.reshape(batch_size, max_seq_length, -1).detach().cpu().numpy()
                true_start_ids = dev_batch_data['start_ids'].detach().cpu().numpy()
                true_end_ids = dev_batch_data['end_ids'].detach().cpu().numpy()
                tmp_start_logits = np.argmax(start_logits, axis=2)
                tmp_end_logits = np.argmax(end_logits, axis=2)
                if s_logits is None:
                    s_logits = tmp_start_logits
                    e_logits = tmp_end_logits
                    true_s_logits = true_start_ids
                    true_e_logits = true_end_ids
                else:
                    s_logits = np.append(s_logits, tmp_start_logits, axis=0)
                    e_logits = np.append(e_logits, tmp_end_logits, axis=0)
                    true_s_logits = np.append(true_s_logits, true_start_ids, axis=0)
                    true_e_logits = np.append(true_e_logits, true_end_ids, axis=0)
            preds = []
            trues = []
            for tmp_s_logits, tmp_e_logits, true_tmp_s_logits, true_tmp_e_logits, tmp_callback_info in zip(s_logits,
                                                                                                           e_logits,
                                                                                                           true_s_logits,
                                                                                                           true_e_logits,
                                                                                                           dev_callback_info):
                text, text_offset, event_type, entities = tmp_callback_info
                tmp_s_logits = tmp_s_logits[text_offset:text_offset + len(text)]
                tmp_e_logits = tmp_e_logits[text_offset:text_offset + len(text)]
                true_tmp_s_logits = true_tmp_s_logits[text_offset:text_offset + len(text)]
                true_tmp_e_logits = true_tmp_e_logits[text_offset:text_offset + len(text)]
                pred_bio = convert_value_to_bio(tmp_s_logits, tmp_e_logits, text, id2rolelabel)
                true_bio = convert_value_to_bio(true_tmp_s_logits, true_tmp_e_logits, text, id2rolelabel)
                preds.append(pred_bio)
                trues.append(true_bio)
            accuracy = accuracy_score(trues, preds)
            precision = precision_score(trues, preds)
            recall = recall_score(trues, preds)
            f1 = f1_score(trues, preds)
            return loss.item(), accuracy, precision, recall, f1

    def test(self, model, model_path):
        model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
        s_logits, e_logits = None, None
        true_s_logits, true_e_logits = None, None
        model.eval()
        with torch.no_grad():
            for eval_step, test_batch_data in enumerate(self.test_loader):
                for key in test_batch_data.keys():
                    test_batch_data[key] = test_batch_data[key].to(device)
                start_logits, end_logits = model(test_batch_data['token_ids'], test_batch_data['attention_masks'],
                                                 test_batch_data['token_type_ids'])
                start_logits = start_logits.detach().cpu().numpy()
                end_logits = end_logits.detach().cpu().numpy()
                true_start_ids = test_batch_data['start_ids'].detach().cpu().numpy()
                true_end_ids = test_batch_data['end_ids'].detach().cpu().numpy()
                tmp_start_logits = np.argmax(start_logits, axis=2)
                tmp_end_logits = np.argmax(end_logits, axis=2)
                if s_logits is None:
                    s_logits = tmp_start_logits
                    e_logits = tmp_end_logits
                    true_s_logits = true_start_ids
                    true_e_logits = true_end_ids
                else:
                    s_logits = np.append(s_logits, tmp_start_logits, axis=0)
                    e_logits = np.append(e_logits, tmp_end_logits, axis=0)
                    true_s_logits = np.append(true_s_logits, true_start_ids, axis=0)
                    true_e_logits = np.append(true_e_logits, true_end_ids, axis=0)
            preds = []
            trues = []
            for tmp_s_logits, tmp_e_logits, true_tmp_s_logits, true_tmp_e_logits, tmp_callback_info in zip(s_logits,
                                                                                                           e_logits,
                                                                                                           true_s_logits,
                                                                                                           true_e_logits,
                                                                                                           test_callback_info):
                text, text_offset, event_type, entities = tmp_callback_info
                tmp_s_logits = tmp_s_logits[text_offset:text_offset + len(text)]
                tmp_e_logits = tmp_e_logits[text_offset:text_offset + len(text)]
                true_tmp_s_logits = true_tmp_s_logits[text_offset:text_offset + len(text)]
                true_tmp_e_logits = true_tmp_e_logits[text_offset:text_offset + len(text)]
                pred_bio = convert_value_to_bio(tmp_s_logits, tmp_e_logits, text, id2rolelabel)
                true_bio = convert_value_to_bio(true_tmp_s_logits, true_tmp_e_logits, text, id2rolelabel)

                preds.append(pred_bio)
                trues.append(true_bio)
            accuracy = accuracy_score(trues, preds)
            precision = precision_score(trues, preds)
            recall = recall_score(trues, preds)
            f1 = f1_score(trues, preds)
            report = classification_report(trues, preds)
            logger.info('[test] accuracy:{} precision:{} recall:{} f1:{}'.format(
                accuracy, precision, recall, f1))
            logger.info(report)

    def predict(self, raw_text, query, model, device, args, query2label):
        model.to(device)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(args.bert_dir, 'vocab.txt'))
            tokens_b = [i for i in raw_text]
            tokens_a = [i for i in query]
            encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                text_pair=tokens_b,
                                                max_length=args.max_seq_len,
                                                padding='max_length',
                                                truncation_strategy='only_second',
                                                return_token_type_ids=True,
                                                return_attention_mask=True)

            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(device)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(device)

            start_logits, end_logits = model(token_ids,
                                             attention_masks,
                                             token_type_ids)

            tmp_start_logits = start_logits.detach().cpu().numpy()
            tmp_end_logits = end_logits.detach().cpu().numpy()
            text_offset = len(tokens_a) + 2
            tmp_start_logits = np.argmax(tmp_start_logits, axis=2)
            tmp_end_logits = np.argmax(tmp_end_logits, axis=2)
            for t_start_logits, t_end_logits in zip(tmp_start_logits, tmp_end_logits):
                temp_start_logits = t_start_logits[text_offset:text_offset + len(text)]
                temp_end_logits = t_end_logits[text_offset:text_offset + len(text)]
                preds = convert_value_to_bio(temp_start_logits, temp_end_logits, raw_text, id2rolelabel)
                preds = get_entities(preds)
                preds = [(pred[0],text[pred[1]:pred[2]+1],pred[1],pred[2]) for pred in preds]
                logger.info(preds)



if __name__ == '__main__':
    final_data_path = args.data_dir + 'final_data/'

    train_features, train_callback_info = commonUtils.read_pkl(final_data_path, 'train')
    train_dataset = dataset.MrcDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)

    dev_features, dev_callback_info = commonUtils.read_pkl(final_data_path, 'dev')
    dev_dataset = dataset.MrcDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)

    test_features, test_callback_info = commonUtils.read_pkl(final_data_path, 'test')
    test_dataset = dataset.MrcDataset(test_features)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,
                             num_workers=2)

    label2id = {}
    id2label = {}
    with open(final_data_path + 'labels.txt', 'r') as fp:
        labels = fp.read().strip().split('\n')
    for i, j in enumerate(labels):
        label2id[j] = i
        id2label[i] = j

    rolelabel2id = {}
    id2rolelabel = {}
    with open(final_data_path + 'rolelabels.txt', 'r') as fp:
        rolelabels = fp.read().strip().split('\n')
    # 将0留出来
    for i, j in enumerate(rolelabels):
        rolelabel2id[j] = i + 1
        id2rolelabel[i + 1] = j
    print(id2rolelabel)
    role_labels = rolelabel2id.keys()
    print(final_data_path + 'labels2query.json')
    with open(final_data_path + 'labels2query.json', 'r') as fp:
        data = fp.read()
        labels2query = eval(data)
    with open(final_data_path + 'labels2rolelabels.json', 'r') as fp:
        labels2rolelabels = json.loads(fp.read())

    model = bertMrc.BertMrcModel(args.bert_dir, args)
    bertForMrc = BertForMrc(model, train_loader, dev_loader, test_loader, args)
    # bertForMrc.train('bertMrc')
    model_path = './checkpoints/bertMrc-4350/model.pt'
    bertForMrc.test(model, model_path)

    model, device = trainUtils.load_model_and_parallel(model, args.gpu_ids, model_path)
    with open(args.data_dir + 'raw_data/' + 'dev.json', 'r') as fp:
        test_data = fp.readlines()
        test_data = random.sample(test_data, 10)
        for i, line in enumerate(test_data):
            raw_dict = eval(line)
            text = raw_dict['text']
            event_list = raw_dict['event_list']
            for event in event_list:
                event_type = event['event_type']
                query = labels2query[event_type]

                logger.info("====================================")
                logger.info("文本：" + text)
                logger.info("预测值：")
                bertForMrc.predict(text, query, model, device, args, id2rolelabel)