import os
import sys
import time
import json
import torch
import random
import gensim
import pyprind
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from importlib import import_module
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append('..')
import utils
from models import BERT_SA


def set_up_seed(seed):
    """
    设置随机种子
    :param seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data():
    """
    加载数据, 并将训练集划分为训练集与验证集
    :return train_X: list
            训练文本
    :return train_y: list
            训练标签
    :return dev_X: list
            验证文本
    :return dev_y: list
            验证标签
    :return test_X: list
            测试文本
    :return test_y: list
            测试标签
    """
    train_path = '../data/train.txt'
    dev_path = '../data/dev.txt'
    test_path = '../data/test.txt'

    train_X, train_y = utils.read_data(train_path)
    dev_X, dev_y = utils.read_data(dev_path)
    test_X, test_y = utils.read_data(test_path)

    return train_X, train_y, dev_X, dev_y, test_X, test_y


def get_labels(y):
    """
    因为 labels 是 str 类型, 要转为 long 类型
    :param y: list
    :return y: list
    """
    return list(map(int, y))


def package_batch(X, y, batch_size):
    """
    将数据封装为 batch
    :param X: list
    :param y: list
    :param batch_size: int
    :return:
    """
    batch_count = int(len(X) / batch_size)
    batch_X, batch_y = [], []
    for i in range(batch_count):
        batch_X.append(X[i * batch_size: (i + 1) * batch_size])
        batch_y.append(y[i * batch_size: (i + 1) * batch_size])

    return batch_X, torch.LongTensor(batch_y), batch_count


def evaluate(model, batch_X, batch_y, batch_count, device):

    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for b in range(batch_count):
            texts = batch_X[b]
            labels = batch_y[b].to(device)

            # shape: (batch, num_outputs)
            pred = model(texts)
            loss = F.cross_entropy(pred, labels)
            loss_total += loss

            # shape: (batch)
            # 因为 pred 是 logits, 需要这样来提取出标签
            predic = torch.max(pred.data, 1)[1]
            labels_all = np.append(labels_all, labels.data.cpu().numpy())
            predict_all = np.append(predict_all, predic.data.cpu().numpy())

    acc = accuracy_score(labels_all, predict_all)
    F1 = f1_score(labels_all, predict_all)
    P = precision_score(labels_all, predict_all)
    R = recall_score(labels_all, predict_all)

    return loss_total / batch_count, acc, P, R, F1


def train(device):
    config = BERT_SA.Config()

    set_up_seed(config.seed)
    train_X, train_y, dev_X, dev_y, test_X, test_y = load_data()
    train_y = get_labels(train_y)
    dev_y = get_labels(dev_y)
    test_y = get_labels(test_y)

    train_batch_X, train_batch_y, train_batch_count = package_batch(train_X,
                                                                    train_y,
                                                                    config.batch_size)
    dev_batch_X, dev_batch_y, dev_batch_count = package_batch(dev_X,
                                                              dev_y,
                                                              config.batch_size)
    test_batch_X, test_batch_y, test_batch_count = package_batch(test_X,
                                                                 test_y,
                                                                 config.batch_size)

    model = BERT_SA.Model(config, device).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    require_improvement = config.early_stop
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    i = 0
    start = time.time()

    print(model)
    for epoch in range(config.num_epochs):
        model.train()
        for b in range(train_batch_count):
            X = train_batch_X[b]
            y = train_batch_y[b].to(device)

            y_hat = model(X)
            optimizer.zero_grad()

            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:

                dev_loss, dev_acc, dev_P, dev_R, dev_F1 = evaluate(model,
                                                                   dev_batch_X,
                                                                   dev_batch_y,
                                                                   dev_batch_count,
                                                                   device)

                print('Epoch %d | Iter %d | dev loss %f | dev accuracy %f | dev precision %f | '
                      'dev recall %f | dev F1 %f' % (
                    epoch + 1, i + 1, dev_loss, dev_acc, dev_P, dev_R, dev_F1
                ))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve = i
                model.train()
                model = model.to(device)

            if i - last_improve > require_improvement:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds args.early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            i += 1
        if flag:
            break

    print('%.2f seconds used' % (time.time() - start))

    model = BERT_SA.Model(config, device).to(device)

    model.load_state_dict(torch.load(config.save_path))
    test_loss, test_acc, test_P, test_R, test_F1 = evaluate(model,
                                                            test_batch_X,
                                                            test_batch_y,
                                                            test_batch_count,
                                                            device)
    print('test loss %f | test accuracy %f | test precision %f | test recall %f | test F1 %f' % (
        test_loss, test_acc, test_P, test_R, test_F1))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(device)
