import os
import numpy as np
import logging
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

from config.config import BilstmForClassificationConfig
from processor.processor import PROCESSOR
from models.bilstmForClassification import BilstmForSequenceClassification
from utils.utils import set_seed, set_logger
from utils.imbalanced import *
from utils.losses import FocalLoss

import warnings

warnings.filterwarnings("ignore")


# logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, criterion, device, config):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.device = device

    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self, train_loader, dev_loader=None):
        total_step = len(train_loader) * self.config.epochs
        self.optimizer, self.scheduler = self.build_optimizers(
            self.model,
            self.config,
            total_step
        )
        global_step = 0
        eval_step = 100
        best_dev_macro_f1 = 0.0
        for epoch in range(1, self.config.epochs + 1):
            for train_step, train_data in enumerate(train_loader):
                self.model.train()
                input_ids, label_ids = train_data
                sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                input_ids = input_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                train_outputs = self.model(input_ids, sentence_lengths)

                loss = self.criterion(train_outputs, label_ids)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if dev_loader:
                    if global_step % eval_step == 0:
                        dev_loss, dev_outputs, dev_targets = self.dev(dev_loader)
                        accuracy, precision, recall, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                        print(
                            "【dev】 loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} macro_f1：{:.4f}".format(
                                dev_loss,
                                accuracy,
                                precision,
                                recall,
                                macro_f1))
                        if macro_f1 > best_dev_macro_f1:
                            checkpoint = {
                                'state_dict': self.model.state_dict(),
                            }
                            best_dev_macro_f1 = macro_f1
                            checkpoint_path = os.path.join(self.config.save_dir, '{}_best.pt'.format(model_name))
                            self.save_ckp(checkpoint, checkpoint_path)

        checkpoint_path = os.path.join(self.config.save_dir, '{}_final.pt'.format(model_name))
        checkpoint = {
            'state_dict': self.model.state_dict(),
        }
        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self, dev_loader):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(dev_loader):
                input_ids, label_ids = dev_data
                sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                input_ids = input_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                outputs = self.model(input_ids, sentence_lengths)

                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(label_ids.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path, test_loader):
        model = self.model
        model = self.load_ckp(model, checkpoint_path)
        model.eval()
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                input_ids, label_ids = test_data
                sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
                input_ids = input_ids.to(self.device)
                label_ids = label_ids.to(self.device)
                outputs = model(input_ids, sentence_lengths)
                loss = self.criterion(outputs, label_ids)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(label_ids.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, checkpoint):
        model = self.model
        model = self.load_ckp(model, checkpoint)
        model.eval()
        with torch.no_grad():
            inputs = [tokenizer["char2id"].get(char, 1) for char in text]
            if len(inputs) >= self.config.max_seq_len:
                input_ids = inputs[:self.config.max_seq_len]
            else:
                input_ids = inputs + [0] * (self.config.max_seq_len - len(inputs))
            input_ids = torch.tensor([input_ids])
            sentence_lengths = torch.sum((input_ids > 0).type(torch.long), dim=-1)
            input_ids = input_ids.to(self.device)
            outputs = model(input_ids, sentence_lengths)

            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=-1).flatten().tolist()
            if len(outputs) != 0:
                outputs = [self.config.id2label[i] for i in outputs]
                return outputs
            else:
                return '不好意思，我没有识别出来'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs, average='macro')
        recall = recall_score(targets, outputs, average='macro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, precision, recall, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels, digits=4)
        return report

    def build_optimizers(self, model, config, t_total):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=config.lr,
                          eps=config.adam_epsilon)
        warmup_steps = int(config.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        return optimizer, scheduler


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(123)
    bilstmForClsConfig = BilstmForClassificationConfig()
    sampling_strategy = bilstmForClsConfig.sampling_strategy
    save_file = open(bilstmForClsConfig.save_metric, 'a')
    save_file.write('====================================\n')
    if bilstmForClsConfig.is_unbalanced_data:
        data_name = "bilstm_unblanced"
        model_name = "bilstm_unblanced"
        train_file = "cnews.train.unbalanced.txt"
        if sampling_strategy == "oversample":
            data_name = "bilstm_oversample"
            model_name = "bilstm_oversample"
            train_file = "cnews.train.randomOverSampler.txt"
        if sampling_strategy == "undersample":
            data_name = "bilstm_undersample"
            model_name = "bilstm_undersample"
            train_file = "cnews.train.randomUnderSampler.txt"
        if sampling_strategy == "datasetsample":
            data_name = "bilstm_datasetsample"
            model_name = "bilstm_datasetsample"
            train_file = "cnews.train.unbalanced.txt"
    else:
        data_name = "bilstm"
        model_name = "bilstm"
        train_file = "cnews.train.txt"

    val_file = "cnews.val.txt"
    test_file = "cnews.test.txt"

    loss_func = bilstmForClsConfig.loss
    if loss_func == "celoss":  # 交叉熵
        criterion = nn.CrossEntropyLoss()
    elif loss_func == "focalloss":
        tmp_label = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
        nums = [5000, 4000, 3000, 2000, 1000, 500, 400, 300, 200, 100]
        label_weight_dict = {i: 1 / j for i, j in zip(tmp_label, nums)}
        ratio = [1 - i / sum(nums) for i in nums]
        label_weight_dict = {i: j for i, j in zip(tmp_label, ratio)}
        weight = [label_weight_dict[i] for i in bilstmForClsConfig.labels]
        criterion = FocalLoss(weight=weight)
    elif loss_func == "weight_celoss":
        tmp_label = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
        nums = [5000, 4000, 3000, 2000, 1000, 500, 400, 300, 200, 100]
        label_weight_dict = {i: 1 / j for i, j in zip(tmp_label, nums)}
        ratio = [1 - i / sum(nums) for i in nums]
        label_weight_dict = {i: j for i, j in zip(tmp_label, ratio)}
        weight = [label_weight_dict[i] for i in bilstmForClsConfig.labels]
        weight = torch.tensor(weight, requires_grad=False).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        raise Exception("请输入正确的损失函数的名称")
    data_name = data_name + "_" + loss_func
    model_name = model_name + "_" + loss_func
    save_file.write("data_name：" + data_name + '\n')
    save_file.write("model_name：" + model_name + '\n')
    save_file.write("loss_function：" + loss_func + '\n')

    # set_logger('{}.log'.format(model_name))

    processor = PROCESSOR['BilstmProcessor']()
    max_seq_len = bilstmForClsConfig.max_seq_len
    tokenizer = {}
    char2id = {}
    id2char = {}
    for i, char in enumerate(bilstmForClsConfig.vocab):
        char2id[char] = i
        id2char[i] = char
    tokenizer["char2id"] = char2id
    tokenizer["id2char"] = id2char

    model = BilstmForSequenceClassification(
        len(bilstmForClsConfig.labels),
        bilstmForClsConfig.vocab_size,
        bilstmForClsConfig.word_embedding_dimension,
        bilstmForClsConfig.hidden_dim
    )
    model = model.to(device)

    trainer = Trainer(model, criterion, device, bilstmForClsConfig)

    if bilstmForClsConfig.do_train:
        train_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, train_file))
        train_dataset = processor.get_examples(
            train_examples,
            max_seq_len,
            tokenizer,
            './data/THUCNews/train_{}.pkl'.format(data_name),
            bilstmForClsConfig.label2id,
            'train')
        if sampling_strategy == "datasetsample":
            train_loader = DataLoader(
                train_dataset,
                sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=callback_get_label),
                batch_size=bilstmForClsConfig.train_batch_size,
                shuffle=False
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=bilstmForClsConfig.train_batch_size, shuffle=True)

        if bilstmForClsConfig.do_eval:
            eval_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, val_file))
            eval_dataset = processor.get_examples(
                eval_examples,
                max_seq_len,
                tokenizer,
                './data/THUCNews/eval_{}.pkl'.format(data_name),
                bilstmForClsConfig.label2id,
                'eval')
            eval_loader = DataLoader(eval_dataset, batch_size=bilstmForClsConfig.eval_batch_size, shuffle=False)
            trainer.train(train_loader, eval_loader)
        else:
            trainer.train(train_loader)

    if bilstmForClsConfig.do_test:
        test_examples = processor.read_data(os.path.join(bilstmForClsConfig.data_dir, test_file))
        test_dataset = processor.get_examples(
            test_examples,
            max_seq_len,
            tokenizer,
            './data/THUCNews/test_{}.pkl'.format(data_name),
            bilstmForClsConfig.label2id,
            'test'
        )
        test_loader = DataLoader(test_dataset, batch_size=bilstmForClsConfig.eval_batch_size, shuffle=False)
        total_loss, test_outputs, test_targets = trainer.test(
            os.path.join(bilstmForClsConfig.save_dir, '{}_final.pt'.format(model_name)),
            test_loader,
        )
        _, _, _, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
        print('macro_f1：{}'.format(macro_f1))
        report = trainer.get_classification_report(test_outputs, test_targets, labels=bilstmForClsConfig.labels)
        print(report)
        save_file.write(report)

    if bilstmForClsConfig.do_predict:
        checkpoint = os.path.join(bilstmForClsConfig.save_dir, '{}_final.pt'.format(model_name))
        with open(os.path.join(bilstmForClsConfig.data_dir, 'cnews.test.txt'), 'r') as fp:
            lines = fp.readlines()
            ind = np.random.randint(0, len(lines))
            line = lines[ind].strip().split('\t')
            text = line[1]
            save_file.write(text + '\n')
            result = trainer.predict(tokenizer, text, checkpoint)
            save_file.write("预测标签：" + str(result) + '\n')
            save_file.write("真实标签：" + line[0] + '\n')
            # print("==========================")
    save_file.write('====================================\n')
