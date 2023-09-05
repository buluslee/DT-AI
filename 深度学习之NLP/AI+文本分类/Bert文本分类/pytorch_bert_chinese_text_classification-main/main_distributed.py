import sys

sys.path.append(r"./")
"""
该文件使用的data_loader.py里面的数据加载方式。
"""
# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,3'
import tempfile
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
import torch.distributed as dist

import bert_config
import models
from utils import utils
from data_loader import CNEWSDataset, Collate, CPWSDataset

logger = logging.getLogger(__name__)


# 单机多卡
# word_size：机器一共有几张卡
# rank：第几块GPU
# local_rank：第几块GPU，和rank相同
# print(torch.cuda.device_count())

# local_rank = torch.distributed.get_rank()
# args.local_rank = local_rank
# print(args.local_rank)
# dist.init_process_group(backend='gloo',
#                         init_method=r"file:///D://Code//project//pytorch-distributed training//tmp",
#                         rank=0,
#                         world_size=1)
# torch.cuda.set_device(args.local_rank)


class Trainer:
    def __init__(self, args, optimizer):
        self.args = args
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss().cuda(self.args.local_rank)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def load_model(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # new_start_dict = {}
        # for k, v in checkpoint['state_dict'].items():
        #     new_start_dict["module." + k] = v
        # model.load_state_dict(new_start_dict)
        model.load_state_dict(checkpoint["state_dict"])
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

    def train(self, model, train_loader, train_sampler, dev_loader=None):
        self.dev_loader = dev_loader
        self.model = model
        total_step = len(train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 10
        best_dev_micro_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            train_sampler.set_epoch(epoch)
            for train_step, train_data in enumerate(train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].cuda(self.args.local_rank)
                attention_masks = train_data['attention_masks'].cuda(self.args.local_rank)
                token_type_ids = train_data['token_type_ids'].cuda(self.args.local_rank)
                labels = train_data['labels'].cuda(self.args.local_rank)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids)

                loss = self.criterion(train_outputs, labels)

                torch.distributed.barrier()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = self.loss_reduce(loss)
                if args.local_rank == 0:
                    logger.info(
                        "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss))
                global_step += 1
                if dev_loader is not None and self.args.local_rank == 0:
                    if global_step % eval_step == 0:
                        dev_loss, dev_outputs, dev_targets = self.dev()
                        accuracy, micro_f1, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                        logger.info(
                            "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss,
                                                                                                       accuracy,
                                                                                                       micro_f1,
                                                                                                       macro_f1))
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
                            self.save_ckp(checkpoint, checkpoint_path)
            if dev_loader is None and self.args.local_rank == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                save_path = os.path.join(self.args.output_dir, args.data_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                checkpoint_path = os.path.join(save_path, 'best.pt')
                self.save_ckp(checkpoint, checkpoint_path)

    def output_reduce(self, outputs, targets):
        output_gather_list = [torch.zeros_like(outputs) for _ in range(self.args.local_world_size)]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(output_gather_list, outputs)

        outputs = torch.cat(output_gather_list, dim=0)
        target_gather_list = [torch.zeros_like(targets) for _ in range(self.args.local_world_size)]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(target_gather_list, targets)
        targets = torch.cat(target_gather_list, dim=0)
        return outputs, targets

    def loss_reduce(self, loss):
        rt = loss.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.local_world_size
        return rt

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].cuda(self.args.local_rank)
                attention_masks = dev_data['attention_masks'].cuda(self.args.local_rank)
                token_type_ids = dev_data['token_type_ids'].cuda(self.args.local_rank)
                labels = dev_data['labels'].cuda(self.args.local_rank)
                outputs = self.model(token_ids, attention_masks, token_type_ids)

                loss = self.criterion(outputs, labels)
                torch.distributed.barrier()
                # total_loss += loss.item()
                loss = self.loss_reduce(loss)
                total_loss += loss
                outputs, targets = self.output_reduce(outputs, labels)
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(targets.cpu().detach().numpy().tolist())
        print(len(dev_outputs), len(dev_targets))
        return total_loss, dev_outputs, dev_targets

    def test(self, model, test_loader):
        model.eval()
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                token_ids = test_data['token_ids'].cuda(self.args.local_rank)
                attention_masks = test_data['attention_masks'].cuda(self.args.local_rank)
                token_type_ids = test_data['token_type_ids'].cuda(self.args.local_rank)
                labels = test_data['labels'].cuda(self.args.local_rank)
                outputs = model(token_ids, attention_masks, token_type_ids)

                loss = self.criterion(outputs, labels)
                torch.distributed.barrier()
                loss = self.loss_reduce(loss)
                total_loss += loss
                outputs, targets = self.output_reduce(outputs, labels)
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(targets.cpu().detach().numpy().tolist())
                # total_loss += loss.item()
                # outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                # test_outputs.extend(outputs.tolist())
                # test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args, model):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.max_seq_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].cuda(self.args.local_rank)
            attention_masks = inputs['attention_mask'].cuda(self.args.local_rank)
            token_type_ids = inputs['token_type_ids'].cuda(self.args.local_rank)
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


def main(args, tokenizer, local_rank, local_world_size):
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    )
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
    if args.local_rank == 0:
        print(label2id)

    args.train_batch_size = int(args.train_batch_size / torch.cuda.device_count())
    args.eval_batch_size = int(args.eval_batch_size / torch.cuda.device_count())

    collate = Collate(tokenizer=tokenizer, max_len=args.max_seq_len, tag2id=label2id)

    train_dataset = dataset(file_path='data/{}/{}'.format(args.data_name, train_file))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              collate_fn=collate.collate_fn, num_workers=4, sampler=train_sampler)
    test_dataset = dataset(file_path='data/{}/{}'.format(args.data_name, test_file))

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             collate_fn=collate.collate_fn, num_workers=4, sampler=test_sampler)
    # test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
    #                          collate_fn=collate.collate_fn, num_workers=4)

    model = models.BertForSequenceClassification(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    trainer = Trainer(args, optimizer)

    if args.do_train:
        if args.retrain:
            checkpoint_path = './checkpoints/{}/best.pt'.format(args.data_name)
            checkpoint = torch.load(checkpoint_path)
            # trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            model.cuda(args.local_rank)
            r_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
            r_model.load_state_dict(checkpoint['state_dict'])
            if args.local_rank == 0:
                logger.info("加载模型继续训练")
            # 训练和验证
            trainer.train(r_model, train_loader, train_sampler, dev_loader=test_loader)
        else:

            model.cuda(args.local_rank)
            r_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
            # 训练和验证
            trainer.train(r_model, train_loader, train_sampler, dev_loader=test_loader)

    # 测试
    if args.do_test:
        if args.local_rank == 0:
            logger.info('========进行测试========')
        checkpoint_path = './checkpoints/{}/best.pt'.format(args.data_name)
        # 多卡预测要先模型并行
        model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        # 再加载模型
        model = trainer.load_model(model, checkpoint_path)

        total_loss, test_outputs, test_targets = trainer.test(model, test_loader)
        accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
        if args.local_rank == 0:
            logger.info(
                "【test】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(total_loss, accuracy,
                                                                                            micro_f1,
                                                                                            macro_f1))
            report = trainer.get_classification_report(test_outputs, test_targets, labels)
            logger.info(report)

    # 预测
    if args.do_predict:
        checkpoint_path = './checkpoints/{}/best.pt'.format(args.data_name)
        model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        model = trainer.load_model(model, checkpoint_path)
        line = test_dataset[0]
        text = line[0]
        result = trainer.predict(tokenizer, text, id2label, args, model)
        if args.local_rank == 0:
            print(text)
            print("预测标签：", result[0])
            print("真实标签：", line[1])
            print("==========================")


def spmd_main(local_world_size, local_rank, init_method):
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    if local_rank == 0:
        for k, v in env_dict.items():
            print(k, v)
    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        if "INIT_METHOD" in os.environ.keys():
            print(f"init_method is {os.environ['INIT_METHOD']}")
            url_obj = urlparse(os.environ["INIT_METHOD"])
            if url_obj.scheme.lower() != "file":
                raise ValueError("Windows only supports FileStore")
            else:
                init_method = os.environ["INIT_METHOD"]
        else:
            # It is a example application, For convience, we create a file in temp dir.
            # current_work_dir = os.getcwd()
            # init_method = f"file:///{os.path.join(current_work_dir, 'ddp_example')}"
            init_method = init_method
        print(init_method)
        dist.init_process_group(backend="gloo", init_method=init_method, rank=int(env_dict["RANK"]),
                                world_size=int(env_dict["WORLD_SIZE"]))
    else:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    main(args, tokenizer, local_rank, local_world_size)

    dist.destroy_process_group()


if __name__ == '__main__':
    # processor = preprocess.Processor()
    args = bert_config.Args().get_parser()
    utils.set_seed(args.seed)
    utils.set_logger(os.path.join(args.log_dir, 'main.log'))
    # The main entry point is called directly without using subprocess
    current_work_dir = os.getcwd()
    init_method = f"file:///{os.path.join(current_work_dir, 'ddp_example')}"
    spmd_main(args.local_world_size, args.local_rank, init_method)
