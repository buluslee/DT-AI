from nltk import word_tokenize
from collections import Counter
import numpy as np
import jieba
import opencc
import os
import torch
import random

from config import Config


def train_test_split(config, test_ratio=0.2, shuffle=True):
    with open(os.path.join(config.data_dir, 'cmn.txt'), 'r') as fp:
        lines = fp.read().strip().split('\n')
    total = len(lines)
    if shuffle:
        random.shuffle(lines)
    train_len = int((1 - test_ratio) * total)
    train_data = lines[:train_len]
    test_data = lines[train_len:]
    with open(os.path.join(config.data_dir, 'train.txt'), 'w') as fp:
        fp.write("\n".join(train_data))
    with open(os.path.join(config.data_dir, 'test.txt'), 'w') as fp:
        fp.write("\n".join(test_data))
    print('总共有数据：{}条'.format(total))
    print('训练集：{}条'.format(len(train_data)))
    print('测试集：{}条'.format(len(test_data)))


def load_file(path, add_begin_end=True):
    """
    读取文件，分别对中文和英文进行分词
    :param path:
    :param add_begin_end:
    :return:
    """
    ens = []
    chs = []
    with open(path, 'r') as fp:
        lines = fp.read().strip().split('\n')
    lines = [line.split('\t') for line in lines]
    for line in lines:
        en = line[0]
        ch = line[1]
        # 繁体转简体
        cc = opencc.OpenCC('t2s')
        ch = cc.convert(ch)
        en_tokens = ['BOS'] + word_tokenize(en.lower()) + ['EOS']
        # 测试时不用加开始和结束标识
        if add_begin_end:
            ch_tokens = ['BOS'] + jieba.lcut(ch, cut_all=False) + ['EOS']
        else:
            ch_tokens = jieba.lcut(ch, cut_all=False)
        # print(en_tokens, ch_tokens)
        ens.append(en_tokens)
        chs.append(ch_tokens)
    return ens, chs


def build_tokenizer(sens, config):
    """
    构建映射字典以及计算vocab_size
    :param sens:
    :param config:
    :return:
    """
    word_count = Counter()
    for sen in sens:
        for word in sen:
            word_count[word] += 1
    ls = word_count.most_common(config.max_vocab_size)
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(ls)}
    word2idx['UNK'] = config.UNK_IDX
    word2idx['PAD'] = config.PAD_IDX

    idx2word = {v: k for k, v in word2idx.items()}
    total_vocab = len(ls) + 2

    return word2idx, idx2word, total_vocab


class Tokenizer(object):
    def __init__(self, word2idx, idx2word, vocab_size):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = vocab_size

    def __str__(self):
        return str(self.word2idx) + \
               '\n' + \
               str(self.idx2word) + \
               '\n' + \
               str(self.vocab_size)


def build_batches(length, batch_size, shuffle=True):
    """
    这里是分batch，每个batch里面存储的是索引
    :param length:
    :param batch_size:
    :param shuffle:
    :return:
    """
    mini_batches = np.arange(0, length, batch_size)
    if shuffle:
        np.random.shuffle(mini_batches)
    result = []
    for idx in mini_batches:
        result.append(np.arange(idx, min(length, idx + batch_size)))

    return result


def prepare_data(seqs):
    """
    将每一个batch里面的数据都填充到里面句子的最大长度
    这里的长度是计算了BOS和EOS的
    :param seqs:
    :return:
    """
    # 处理每个batch句子（一个batch中句子长度可能不一致，需要pad）
    batch_size = len(seqs)
    lengthes = [len(seq) for seq in seqs]  # 每个句子的长度列表

    max_length = max(lengthes)  # 句子最大长度
    # 初始化句子矩阵都为0
    x = np.zeros((batch_size, max_length)).astype("int32")
    for idx in range(batch_size):
        # 按行将每行句子赋值进去
        x[idx, :lengthes[idx]] = seqs[idx]

    x_lengths = np.array(lengthes).astype("int32")
    return x, x_lengths


class DataProcessor:
    def __init__(self, config):
        cached_en_tokenizer = os.path.join(config.data_dir, "cached_{}".format("en_tokenizer"))
        cached_cn_tokenizer = os.path.join(config.data_dir, "cached_{}".format("cn_tokenizer"))
        if not os.path.exists(cached_en_tokenizer) or not os.path.exists(cached_cn_tokenizer):
            ens, chs = load_file(os.path.join(config.data_dir, 'cmn.txt'))
            en_word2idx, en_idx2word, en_vocab_size = build_tokenizer(ens, config)
            ch_word2idx, ch_idx2word, ch_vocab_size = build_tokenizer(chs, config)
            torch.save([en_word2idx, en_idx2word, en_vocab_size], cached_en_tokenizer)
            torch.save([ch_word2idx, ch_idx2word, ch_vocab_size], cached_cn_tokenizer)
        else:
            en_word2idx, en_idx2word, en_vocab_size = torch.load(cached_en_tokenizer)
            ch_word2idx, ch_idx2word, ch_vocab_size = torch.load(cached_cn_tokenizer)
        self.en_tokenizer = Tokenizer(en_word2idx, en_idx2word, en_vocab_size)
        self.ch_tokenizer = Tokenizer(ch_word2idx, ch_idx2word, ch_vocab_size)
        print(self.en_tokenizer)
        print(self.ch_tokenizer)

    def get_train_examples(self, config):
        return self.create_examples(os.path.join(config.data_dir, 'train.txt'), 'train', config)

    def get_dev_examples(self, config):
        return self.create_examples(os.path.join(config.data_dir, "test.txt"), "dev", config)

    def create_examples(self, path, set_type, config, sort_reverse=True):
        en_sents, ch_sents = load_file(path)
        # 转换为对应的id
        out_en_sents = [[self.en_tokenizer.word2idx.get(word, config.UNK_IDX) for word in sen] for sen in en_sents]
        out_ch_sents = [[self.ch_tokenizer.word2idx.get(word, config.UNK_IDX) for word in sen] for sen in ch_sents]
        if sort_reverse:
            sorted_index = self.sort_sents(out_en_sents)
            out_en_sents = [out_en_sents[idx] for idx in sorted_index]
            out_ch_sents = [out_ch_sents[idx] for idx in sorted_index]
        # 这里是英译中，所以我们根据英文句子的长度构建mini_batch
        mini_batches = build_batches(len(out_en_sents), config.batch_size)
        all_examples = []
        for mini_batch in mini_batches:
            mb_en_sentences = [out_en_sents[i] for i in mini_batch]
            mb_ch_sentences = [out_ch_sents[i] for i in mini_batch]

            mb_x, mb_x_len = prepare_data(mb_en_sentences)
            mb_y, mb_y_len = prepare_data(mb_ch_sentences)

            all_examples.append((mb_x, mb_x_len, mb_y, mb_y_len))

        return all_examples

    def sort_sents(self, sents):
        """
        按照长度进行排序
        :param sents:
        :return:
        """
        return sorted(range(len(sents)), key=lambda x: len(sents[x]), reverse=True)


if __name__ == '__main__':
    # 划分训练集和测试集，可跳过
    # train_test_split(Config)
    dataProcessor = DataProcessor(Config)
    tran_examples = dataProcessor.get_train_examples(Config)
    print(tran_examples[0])
    dev_examples = dataProcessor.get_dev_examples(Config)
    print(dev_examples[0])
