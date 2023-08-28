import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.save_path = '../model_parameters/CNN_SA.bin'

        # 模型配置
        self.filter_sizes = (3, 4, 5)
        self.num_filters = 100
        self.dense_hidden_size = 128
        self.dropout = 0.5
        self.embed_size = 300
        self.num_outputs = 2


class Model(nn.Module):
    def __init__(self, embed, config):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.ffn = nn.Linear(config.num_filters * len(config.filter_sizes), config.dense_hidden_size)
        self.classifier = nn.Linear(config.dense_hidden_size, config.num_outputs)

    def max_pooling(self, x):
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    # def conv_and_pool(self, x, conv):
    #     x = F.relu(conv(x)).squeeze(3)
    #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    def forward(self, inputs):
        # shape: (batch_size, max_seq_length, embed_size)
        embed = self.embedding(inputs)

        # CNN 接受四维数据输入,
        # 第一维: batch,
        # 第二维: 通道数 (Channel), 在图像中指的是 RGB 这样的通道, 在自然语言里面指的是多少种词嵌入, 本项目中仅采用一种词嵌入, 所以就是 1 通道
        # 第三维: 高度 (Height), 在图像中指的是图片的高, 在自然语言里面就是序列长度
        # 第四维: 宽度 (Weight), 在图像中指的是图片的宽, 在自然语言里面就是嵌入维度
        # shape: (batch_size, 1, max_seq_length, embed_size)
        embed = embed.unsqueeze(1)

        cnn_outputs = []
        for conv in self.convs:
            # shape: (batch_size, filter_size, max_seq_length - kernel_size + 1, 1)
            conv_output = conv(embed)
            # shape: (batch_size, filter_size, max_seq_length - kernel_size + 1, 1)
            relu_output = self.relu(conv_output)
            # shape: (batch_size, filter_size, max_seq_length - kernel_size + 1, 1)
            relu_output = relu_output.squeeze(3)
            # shape: (batch_size, filter_size)
            pooling_output = self.max_pooling(relu_output)
            cnn_outputs.append(pooling_output)

        # cnn_outputs = torch.cat([self.conv_and_pool(embed, conv) for conv in self.convs], 1)

        # shape: (batch, num_filters * len(num_filters))
        cnn_outputs = torch.cat(cnn_outputs, 1)
        cnn_outputs = self.dropout(cnn_outputs)
        # shape: (batch, dense_hidden_size)
        ffn_output = self.relu(self.ffn(cnn_outputs))
        # shape: (batch, num_outputs)
        logits = self.classifier(ffn_output)

        return logits
