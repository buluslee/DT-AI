import os

import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.callback_get_label = callback_get_label
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)  # 每一个样本的label
        df.index = self.indices  # 每一个样本的索引
        df = df.sort_index()  # 按索引对标签排序

        label_to_count = df["label"].value_counts()  # 统计每一类的数量
        weights = 1.0 / label_to_count[df["label"]]  # 计算每一个样本被采样的权重
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def callback_get_label(dataset):
    # 把标签张量转换为Python的整形
    return [int(i[1]) for i in dataset]


if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from processor.processor import PROCESSOR

    processor = PROCESSOR['BilstmProcessor']()
    data_dir = "../data/THUCNews/"
    train_file = "cnews.train.unbalanced.txt"
    data_name = 'bilstm_datasetSampler'
    max_seq_len = 256
    with open(os.path.join(data_dir, 'cnews.vocab.txt'), 'r') as fp:
        vocab = fp.read().strip().split('\n')
    vocab_size = len(vocab)

    tokenizer = {}
    char2id = {}
    id2char = {}
    for i, char in enumerate(vocab):
        char2id[char] = i
        id2char[i] = char
    tokenizer["char2id"] = char2id
    tokenizer["id2char"] = id2char

    with open(os.path.join(data_dir, 'cnews.labels'), 'r') as fp:
        labels = fp.read().strip().split('\n')
    label2id = {}
    id2label = {}
    for k, v in enumerate(labels):
        label2id[v] = k
        id2label[k] = v

    print(os.path.join(data_dir, train_file))
    train_examples = processor.read_data(os.path.join(data_dir, train_file))
    train_dataset = processor.get_examples(
        train_examples,
        max_seq_len,
        tokenizer,
        '../data/THUCNews/train_{}.pkl'.format(data_name),
        label2id,
        'train')
    for i in train_dataset:
        print(i)
        break

    # 这里的batch_size会影响到sampler
    train_loader = DataLoader(
        train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=callback_get_label),
        batch_size=32,
        shuffle=False
    )

    for step, data in enumerate(train_loader):
        print(step, data[0].shape, data[1].shape)
        break
