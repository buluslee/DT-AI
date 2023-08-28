"""
:author: Qizhi Li
"""
import torch
import torch.nn as nn


class Config:
    def __init__(self):
        # 训练配置
        self.seed = 22
        self.batch_size = 64
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.num_epochs = 100
        self.early_stop = 512
        self.max_seq_length = 128
        self.save_path = '../model_parameters/BiLSTM_SA.bin'

        # 模型配置
        self.lstm_hidden_size = 128
        self.dense_hidden_size = 128
        self.embed_size = 300
        self.num_layers = 1
        self.num_outputs = 2


class Model(nn.Module):

    def __init__(self, embed, config):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
        self.LSTM = nn.LSTM(config.embed_size, config.lstm_hidden_size,
                            num_layers=config.num_layers, batch_first=True,
                            bidirectional=True)
        # 因为是双向 LSTM, 所以要乘2
        self.ffn = nn.Linear(config.lstm_hidden_size * 2,
                             config.dense_hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(config.dense_hidden_size,
                                    config.num_outputs)

    def forward(self, inputs):
        # shape: (batch_size, max_seq_length, embed_size)
        embed = self.embedding(inputs)
        # shape: (batch_size, max_seq_length, lstm_hidden_size * 2)
        lstm_hidden_states, _ = self.LSTM(embed)
        # LSTM 的最后一个时刻的隐藏状态, 即句向量
        # shape: (batch, lstm_hidden_size * 2)
        lstm_hidden_states = lstm_hidden_states[:, -1, :]
        # shape: (batch, dense_hidden_size)
        ffn_outputs = self.relu(self.ffn(lstm_hidden_states))
        # shape: (batch, num_outputs)
        logits = self.classifier(ffn_outputs)

        return logits
