import os
import logging
import numpy as np
import json
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AutoTokenizer

import config
import data_loader
import globalpoint
from utils.common_utils import set_seed, set_logger, read_json, trans_ij2k, fine_grade_tokenize
from utils.train_utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model
from utils.metric_utils import calculate_metric, classification_report, get_p_r_f
from tensorboardX import SummaryWriter

args = config.Args().get_parser()
set_seed(args.seed)
logger = logging.getLogger(__name__)

if args.use_tensorboard == "True":
    writer = SummaryWriter(log_dir='./tensorboard')


class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag, model, device):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        self.model = model
        self.device = device
        if train_loader is not None:
            self.t_total = len(self.train_loader) * args.train_epochs
            self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = self.args.eval_steps  # 每多少个step打印损失及进行验证
        best_f1 = 0.0
        for epoch in range(1, self.args.train_epochs+1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data:
                    batch = batch.to(self.device)
                loss, logits = self.model(batch_data[0], batch_data[1], batch_data[2], batch_data[3])

                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))

                global_step += 1
                if self.args.use_tensorboard == "True":
                    writer.add_scalar('data/loss', loss.item(), global_step)
                if global_step % eval_steps == 0:
                   dev_loss, precision, recall, f1_score = self.dev()
                   logger.info('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(dev_loss, precision, recall, f1_score))
                   if f1_score > best_f1:
                       save_model(self.args, self.model, model_name, global_step)
                       best_f1 = f1_score
        logger.info("best f1:{}".format(best_f1))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            pred_entities = []
            true_entities = []
            tot_dev_loss = 0.0
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                labels = dev_batch_data[3]
                for dev_batch in dev_batch_data:
                    dev_batch = dev_batch.to(device)
                # logits:[8, 8, 150, 150]
                _, logits = model(dev_batch_data[0], dev_batch_data[1], dev_batch_data[2], dev_batch_data[3])
                batch_size = logits.size(0)
                dev_callbak = dev_callback[eval_step * batch_size:(eval_step + 1) * batch_size]

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i, ...]
                    tokens = dev_callbak[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() > 0.5)):
                            pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    pred_entities.append(dict(pred_tmp))

                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i, ...]
                    tokens = dev_callbak[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() == 1)):
                            true_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            total_count = [0 for _ in range(len(id2tag))]
            role_metric = np.zeros([len(id2tag), 3])
            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(id2tag), 3])
                for idx, _type in enumerate(label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])

                role_metric += tmp_metric

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            # print('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
            return tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]

    def test(self, model_path):
        model = globalpoint.GlobalPointerNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        pred_entities = []
        true_entities = []
        with torch.no_grad():
            for eval_step, dev_batch_data in enumerate(dev_loader):
                labels = dev_batch_data[3]
                for dev_batch in dev_batch_data:
                    dev_batch = dev_batch.to(device)
                # logits:[8, 8, 150, 150]
                _, logits = model(dev_batch_data[0], dev_batch_data[1], dev_batch_data[2], dev_batch_data[3])
                batch_size = logits.size(0)
                dev_callbak = dev_callback[eval_step * batch_size:(eval_step + 1) * batch_size]

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i, ...]
                    tokens = dev_callbak[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() > 0.5)):
                            pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    pred_entities.append(dict(pred_tmp))

                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i, ...]
                    tokens = dev_callbak[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() == 1)):
                            true_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            total_count = [0 for _ in range(len(id2tag))]
            role_metric = np.zeros([len(id2tag), 3])
            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(id2tag), 3])
                for idx, _type in enumerate(label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])

                role_metric += tmp_metric
            logger.info(classification_report(role_metric, label_list, id2tag, total_count))

    def predict(self, raw_text, model_path):
        model = globalpoint.GlobalPointerNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(self.args.bert_dir)
            # tokens = fine_grade_tokenize(raw_text, tokenizer)
            tokens = [i for i in raw_text]
            encode_dict = tokenizer.encode_plus(text=tokens,
                                                max_length=self.args.max_seq_len,
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
              for j in range(self.args.num_tags):
                  logit_ = logit[j, :len(tokens), :len(tokens)]
                  for start, end in zip(*np.where(logit_.cpu().numpy() > 0.5)):
                      pred_tmp[id2tag[j]].append(["".join(tokens[start:end + 1]), start-1])

            logger.info(dict(pred_tmp))


if __name__ == '__main__':

    if args.use_efficient_globalpointer == "True":
      model_name = 'bert-1-eff'
    else:
      model_name = 'bert-1'

    set_logger(os.path.join(args.log_dir, '{}.log'.format(model_name)))
    data_path = os.path.join(args.data_dir, 'mid_data')
    label_list = read_json(data_path, 'labels')
    tag2id = {}
    id2tag = {}
    for k, v in enumerate(label_list):
        tag2id[v] = k
        id2tag[k] = v

    logger.info(args)
    max_seq_len = args.max_seq_len
    if "guwenbert" in args.bert_dir:
      tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    else:
      tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = globalpoint.GlobalPointerNer(args)
    model, device = load_model_and_parallel(model, args.gpu_ids)


    collate = data_loader.Collate(max_len=max_seq_len, tag2id=tag2id, device=device)

    train_dataset, _ = data_loader.MyDataset(file_path=os.path.join(data_path, 'train.json'),
                                              tokenizer=tokenizer,
                                              max_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=collate.collate_fn)
    dev_dataset, dev_callback = data_loader.MyDataset(file_path=os.path.join(data_path, 'dev.json'),
                                                      tokenizer=tokenizer,
                                                      max_len=max_seq_len)

    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate.collate_fn)

    test_dataset, test_callback = data_loader.MyDataset(file_path=os.path.join(data_path, 'test.json'),
                                                      tokenizer=tokenizer,
                                                      max_len=max_seq_len)

    test_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate.collate_fn)

    bertForNer = BertForNer(args, train_loader, dev_loader, test_loader, id2tag, model, device)
    bertForNer.train()

    with open("./checkpoints/{}/args.json".format(model_name), "w", encoding="utf-8") as fp:
      json.dump(vars(args), fp, ensure_ascii=False)
    

    model_path = './checkpoints/{}/model.pt'.format(model_name)
    bertForNer.test(model_path)

    if "cner" in args.data_dir:
      raw_text = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"
    elif "guwen" in args.data_dir:
      raw_text = "冬十月，天子拜太祖兖州牧。十二月，雍丘溃，超自杀。夷邈三族。邈诣袁术请救，为其众所杀，兖州平，遂东略陈地。"
      
    logger.info(raw_text)
    bertForNer.predict(raw_text, model_path)
