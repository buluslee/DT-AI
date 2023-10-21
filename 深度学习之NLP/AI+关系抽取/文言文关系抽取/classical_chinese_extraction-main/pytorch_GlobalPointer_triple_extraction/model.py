import math
import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel

class SparseMultilabelCategoricalCrossentropy(nn.Module):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
    """
    def __init__(self, mask_zero=False, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.mask_zero = mask_zero
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if self.mask_zero:
            infs = zeros + float('inf')
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if self.mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, dim=-1, index=y_true)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)  # a
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # b-a
        aux_loss = torch.clamp(1 - torch.exp(aux_loss), self.epsilon, 1)  # 1-exp(b-a)
        neg_loss = all_loss + torch.log(aux_loss)  # a + log[1-exp(b-a)]
        return pos_loss + neg_loss


class MyLoss(SparseMultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
    def forward(self, y_preds, y_trues):
        ''' y_preds: [Tensor], shape为[btz, heads, seq_len ,seq_len]
        '''
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            shape = y_pred.shape
            # 乘以seq_len是因为(i, j)在展开到seq_len*seq_len维度对应的下标是i*seq_len+j
            y_true = y_true[..., 0] * shape[2] + y_true[..., 1]  # [btz, heads, 实体起终点的下标]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))  # [btz, heads, seq_len*seq_len]
            loss = super().forward(y_pred, y_true.long())
            loss = torch.mean(torch.sum(loss, dim=1))
            loss_list.append(loss)
        return {'loss': sum(loss_list)/3, 'entity_loss': loss_list[0], 'head_loss': loss_list[1], 'tail_loss': loss_list[2]}


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''Returns: [seq_len, d_hid]
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """
    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)
    
    def forward(self, qw, seq_dim=-2):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """
    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' inputs: [..., hdsz]
            mask: [bez, seq_len], padding部分为0
        '''
        sequence_output = self.p_dense(inputs)  # [..., head_size*2]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len]
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2)  # [btz, head_size, seq_len,2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, head_size, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """
    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask

        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

        

    def forward(self, inputs, mask=None):
        ''' inputs: [..., hdsz]
            mask: [bez, seq_len], padding部分为0
        '''
        # [batchsize, 150, 8*64*2]
        sequence_output = self.dense(inputs)  # [..., heads*head_size*2]
        # torch.chunk(sequence_output, self.heads, dim=-1) 8个(batchsize, 150, 64*2)
        # [batchsize, 150, 8, 64*2]
        sequence_output = torch.stack(torch.chunk(sequence_output, self.heads, dim=-1), dim=-2)  # [..., heads, head_size*2]
        # qw:[batchsize, 150, 8, 64], kw:[batchsize, 150, 8, 64]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., heads, head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits / self.head_size**0.5


class GlobalPointerRe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True,
                            hidden_dropout_prob=args.dropout_prob)
        self.entity_output = GlobalPointer(hidden_size=768, heads=2, head_size=64)
        self.head_output = GlobalPointer(hidden_size=768, heads=args.num_tags, head_size=64, RoPE=False, tril_mask=False)
        self.tail_output = GlobalPointer(hidden_size=768, heads=args.num_tags, head_size=64, RoPE=False, tril_mask=False)
        self.criterion = MyLoss(mask_zero=True)

    def forward(self, 
          token_ids, 
          attention_masks, 
          token_type_ids,
          head_labels=None,
          tail_labels=None,
          entity_labels=None):
        bert_output = self.bert(token_ids, attention_masks, token_type_ids)  # [btz, seq_len, hdsz]
        hidden_states = bert_output[0]
        mask = attention_masks

        entity_output = self.entity_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        head_output = self.head_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        tail_output = self.tail_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        if head_labels is None:
          return entity_output, head_output, tail_output
        loss = self.criterion([entity_output, head_output, tail_output], [entity_labels, head_labels, tail_labels])
        return loss