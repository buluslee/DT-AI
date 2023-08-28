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
from models import BiLSTM_SA


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


def build_vocab(vocab):
    """
    构建词表
    简单来说, 就是将文本用 '空格' 切词, 然后放到集合中
    :param train_X: list
            训练文本
    :param dev_X: list
            验证文本
    :param test_X: list
            测试文本
    :return vocab: set
            词表
    :return word2id: dict
            词语与 id 的映射
    :return id2word: dict
            id 与词语的映射
    """
    wv_path = '../data/GoogleNews-vectors-negative300.bin'
    wv = gensim.models.KeyedVectors.load_word2vec_format(wv_path, binary=True)
    word2id, id2word = {}, {}

    i = 0
    pper = pyprind.ProgPercent(len(vocab))
    for word in vocab:
        word = word.rstrip('\n')
        word2id[word] = i
        id2word[word] = i

        try:
            word_embedding = wv[word]
        except KeyError:
            word_embedding = wv['UNK']

        word_embedding = word_embedding.reshape(1, -1)
        if i == 0:
            embeddings = word_embedding
        else:
            embeddings = np.concatenate((embeddings, word_embedding), axis=0)

        i += 1
        pper.update()

    embeddings = np.concatenate((embeddings, wv['UNK'].reshape(1, -1)), axis=0)
    word2id['UNK'] = i
    id2word[i] = 'UNK'

    embeddings = np.concatenate((embeddings, wv['PAD'].reshape(1, -1)), axis=0)
    word2id['PAD'] = i + 1
    id2word[i + 1] = 'PAD'

    print(embeddings.shape)
    print(len(word2id))

    np.save('../data/vocab.npy', embeddings)
    with open('../data/word2id.json', 'w') as f:
        json.dump(word2id, f)

    with open('../data/id2word.json', 'w') as f:
        json.dump(id2word, f)

    return embeddings, word2id, id2word


def turn_sentence_to_id(input_sentence, word2id, max_seq_length):
    """
    将输入的文本变为 id
    :param input_sentence: list
            输入文本
    :param word2id: dict
            词语与 id 的映射关系
    :param max_seq_length: int
            最大句子长度
    :return sentence_ids: LongTensor
            将所有词语转换为了 id
    """
    sentence_ids = []
    pper = pyprind.ProgPercent(len(input_sentence))
    for sentence in input_sentence:
        words = word_tokenize(sentence)
        # 先将句子用 PAD 填充至最大句子长度, 再替换 PAD 为词语 id
        sentence_id = [word2id['PAD']] * max_seq_length
        for i, word in enumerate(words):
            if i >= max_seq_length:
                # 截断
                break

            word = word.lower()  # 因为词表里面的单词都是小写的
            try:
                sentence_id[i] = word2id[word]
            except KeyError:
                sentence_id[i] = word2id['UNK']

        sentence_ids.append(sentence_id)
        pper.update()

    return torch.LongTensor(sentence_ids)


def get_labels(y):
    """
    因为 labels 是 str 类型, 要转为 long 类型, 再转为 LongTensor
    :param y: list
    :return y: LongTensor
    """
    y = list(map(int, y))

    return torch.LongTensor(y)


def evaluate(model, data_loader, device):

    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = texts.to(device)
            labels = labels.to(device)

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

    return loss_total / len(data_loader), acc, P, R, F1


def train(args, device):
    model_name = args.model  # 'BiLSTM_SA, CNN_SA'
    module = import_module('models.' + model_name)
    config = module.Config()

    # config = BiLSTM_SA.Config()

    set_up_seed(config.seed)
    train_X, train_y, dev_X, dev_y, test_X, test_y = load_data()

    if os.path.exists('../data/word2id.json'):
        with open('../data/word2id.json', 'r') as f:
            word2id = json.load(f)

        with open('../data/id2word.json', 'r') as f:
            id2word = json.load(f)

        embeddings = np.load('../data/vocab.npy')
    else:
        with open(r'../data/imdb.vocab', 'r') as f:
            vocab = f.readlines()

        embeddings, word2id, id2word = build_vocab(vocab)

    # 将所有句子给转为 id
    print('Start processing train_X')
    train_X = turn_sentence_to_id(train_X, word2id, config.max_seq_length)
    print('Start processing dev_X')
    dev_X = turn_sentence_to_id(dev_X, word2id, config.max_seq_length)
    print('Start processing test_X')
    test_X = turn_sentence_to_id(test_X, word2id, config.max_seq_length)

    # 将标签转为 LongTensor
    train_y = get_labels(train_y)
    dev_y = get_labels(dev_y)
    test_y = get_labels(test_y)

    train_dataset = Data.TensorDataset(train_X, train_y)
    dev_dataset = Data.TensorDataset(dev_X, dev_y)
    test_dataset = Data.TensorDataset(test_X, test_y)

    train_loader = Data.DataLoader(train_dataset, batch_size=config.batch_size)
    dev_loader = Data.DataLoader(dev_dataset, batch_size=config.batch_size)
    test_loader = Data.DataLoader(test_dataset, batch_size=config.batch_size)

    embeddings = torch.from_numpy(embeddings)
    model = module.Model(embeddings, config).to(device)

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
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            optimizer.zero_grad()

            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:

                dev_loss, dev_acc, dev_P, dev_R, dev_F1 = evaluate(model, dev_loader, device)

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

    model = module.Model(embeddings, config).to(device)

    model.load_state_dict(torch.load(config.save_path))
    test_loss, test_acc, test_P, test_R, test_F1 = evaluate(model, test_loader, device)
    print('test loss %f | test accuracy %f | test precision %f | test recall %f | test F1 %f' % (
        test_loss, test_acc, test_P, test_R, test_F1))


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='choose a model: BiLSTM_SA, CNN_SA', default='BiLSTM_SA')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(args, device)
