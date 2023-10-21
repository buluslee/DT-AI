import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(config.enc_vocab_size, config.embed_size)

        self.rnn = nn.GRU(config.embed_size, config.enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        # 将encoder的输出转为decoder的输入，* 2 是使用了bidirectional
        self.fc = nn.Linear(config.enc_hidden_size * 2, config.dec_hidden_size)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embed(x))

        # 新版pytorch增加了batch里的排序功能，默认需要强制倒序
        # pack_padded_sequence里面的length需要放在cpu上
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        # hid 【2, batch, enc_hidden_size】
        packed_out, hid = self.rnn(packed_embedded)
        # 【batch, seq, 2 * enc_hidden_size】
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max(lengths))

        # 将hid双向叠加 【batch, 2*enc_hidden_size】
        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        # 转为decoder输入hidden state 【1,batch,dec_hidden_size】
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid


class Attention(nn.Module):
    """  """

    def __init__(self, config):
        super(Attention, self).__init__()

        self.enc_hidden_size = config.enc_hidden_size
        self.dec_hidden_size = config.dec_hidden_size

        self.liner_in = nn.Linear(2 * config.enc_hidden_size, config.dec_hidden_size)
        self.liner_out = nn.Linear(2 * config.enc_hidden_size + config.dec_hidden_size, config.dec_hidden_size)

    def forward(self, output, context, mask):
        # context 上下文输出，即encoder的gru hidden state 【batch,enc_seq,enc_hidden*2】
        # output  decoder的gru hidden state  【batch,dec_seq, dec_hidden】
        # mask 【batch, dec_seq, enc_seq】mask在decoder中创建

        batch_size = context.shape[0]
        enc_seq = context.shape[1]
        dec_seq = output.shape[1]

        # score计算公式使用双线性模型 h*w*s
        context_in = self.liner_in(context.reshape(batch_size * enc_seq, -1).contiguous())
        context_in = context_in.view(batch_size, enc_seq, -1).contiguous()
        atten = torch.bmm(output, context_in.transpose(1, 2))
        # 【batch,dec_seq,enc_seq】

        atten.data.masked_fill(mask, -1e6)  # mask置零
        atten = F.softmax(atten, dim=2)

        # 将score和value加权求和，得到输出
        # 【batch, dec_seq, 2*enc_hidden】
        context = torch.bmm(atten, context)
        # 将attention + output 堆叠获取融合信息
        output = torch.cat((context, output), dim=2)

        # 最终输出 batch,dec_seq,dec_hidden_size
        output = torch.tanh(self.liner_out(output.view(batch_size * dec_seq, -1))).view(batch_size, dec_seq, -1)

        return output, atten


class Decoder(nn.Module):
    """"""

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.dec_vocab_size, config.embed_size)
        self.atten = Attention(config)
        # decoder不使用bidirectional
        self.rnn = nn.GRU(config.embed_size, config.dec_hidden_size, batch_first=True)
        self.out = nn.Linear(config.dec_hidden_size, config.dec_vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def create_mask(self, x_len, y_len):
        # 最长句子的长度
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        # 句子batch
        batch_size = len(x_len)

        # 将超出自身序列长度的元素设为False
        x_mask = (torch.arange(max_x_len.item())[None, :].to(self.config.device) < x_len[:, None]).float()  # [batch,max_x_len]
        y_mask = (torch.arange(max_y_len.item())[None, :].to(self.config.device) < y_len[:, None]).float()  # [batch,max_y_len]

        # y_mask[:, :, None] size: [batch,max_y_len,1]
        # x_mask[:, None, :] size:  [batch,1,max_x_len]
        # 需要mask的地方设置为true
        mask = (1 - y_mask[:, :, None] * x_mask[:, None, :]) != 0
        # [batch_size, max_y_len, max_x_len]
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        '''
        :param ctx:encoder层的输出 ： 【batch, enc_seq, 2*enc_hidden】
        :param ctx_lengths: encoder层输入句子的长度list
        :param y: decoder层的输入 【batch, dec_seq, dec_hidden】
        :param y_lengths: decoder输入的句子长度
        :param hid: encoder层输出的最后一个hidden state 【1, batch, dec_hidden】
        :return:
        '''
        y_embed = self.dropout(self.embed(y))
        # 这里没法保证译文也是排倒序
        y_packed = nn.utils.rnn.pack_padded_sequence(y_embed, y_lengths.cpu(), batch_first=True, enforce_sorted=False)
        # 将emcoder的hidden state作为decoder的第一个hidden state
        pack_output, hid = self.rnn(y_packed, hid)
        output_seq, _ = nn.utils.rnn.pad_packed_sequence(pack_output, batch_first=True, total_length=max(y_lengths))

        # 做attention之前需要创建mask
        mask = self.create_mask(ctx_lengths, y_lengths)
        # annention处理
        output, atten = self.atten(output_seq, ctx, mask)
        # 将输出转为vocab_size的softmax概率分布并取对数
        output = F.log_softmax(self.out(output), dim=-1)

        return output, atten, hid


class Seq2Seq(nn.Module):
    '''
        模型架构
    '''

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        context, hid = self.encoder(x, x_lengths)

        output, atten, hid = self.decoder(
            context, x_lengths,
            y, y_lengths,
            hid
        )
        # output: 【batch,output_len,vocab_size】
        # atten   【batch,output_len,input_len】
        return output, atten

    def beam_search(self, x, x_lengths, y, EOS_id, topk=5, max_length=100):
        # hid: [1,1,512]，第二位为batchsize
        encoder_out, hid = self.encoder(x, x_lengths)
        # batch_size = x.shape[0]
        # preds = []
        # attens = []
        # for i in range(max_length):
        #
        #     output,atten,hid = self.decoder(
        #         encoder_out,x_lengths,
        #         y,torch.ones(batch_size).long().to(y.device),
        #         hid
        #     )
        #     # 取出预测概率最大index
        #     y = output.argmax(2).view(batch_size,1)
        #     preds.append(y)
        #     attens.append(atten)

        BOS_id = y[0][0].item()  # 2
        hypotheses = [[BOS_id]]  # [[2]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=y.device)
        completed_hypotheses = []
        t = 0
        while len(completed_hypotheses) < topk and t < max_length:
            t += 1
            hyp_num = len(hypotheses)  # 初始值为1
            # 扩展成batch
            # [1, x_lengths, 1024]
            exp_src_encodings = encoder_out.expand(hyp_num, encoder_out.shape[1], encoder_out.shape[2])
            exp_x_lengths = x_lengths.expand(hyp_num)  # [x_lengths]
            exp_hid = hid.expand(hid.shape[0], hyp_num, hid.shape[2])  # [1,1,512]
            output_t, atten_t, exp_hid = self.decoder(
                exp_src_encodings, exp_x_lengths,
                torch.tensor(hypotheses).long().to(y.device), torch.ones(hyp_num).long().to(y.device) * t,
                exp_hid
            )
            live_hyp_num = topk - len(completed_hypotheses)

            # 这里把num * vocab 展开来方便取topk
            contiuating_hyp_scores = (
                    hyp_scores.unsqueeze(1).expand(hyp_num, output_t.shape[-1]) + output_t[:, -1, :].squeeze(
                1)).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
            # 标记当前概率最大的k个，其是跟在哪个单词的后面
            prev_hyp_ids = top_cand_hyp_pos // (output_t.shape[-1])
            hyp_word_ids = top_cand_hyp_pos % (output_t.shape[-1])

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                # 将当前最大概率的k个，拼接在正确的prev单词后面
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word_id]
                if hyp_word_id == EOS_id:
                    # 搜寻终止
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == topk:
                break

            hypotheses = new_hypotheses
            # print(hypotheses)
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=y.device)

        # 若搜寻了max_len后还没有一个到达EOS则取第一个
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        '''损失函数'''
        super(LanguageModelCriterion, self).__init__()

    def forward(self, inuptY, target, mask):
        # inputY: batch, seq_len, vocab_size
        # target/mask: batch, seq_len
        inuptY = inuptY.contiguous().view(-1, inuptY.shape[2])
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        # 模型seq2seq的输出已经经过log-softmax了，只需将target对应index值收集后在mask
        output = - inuptY.gather(1, target) * mask
        return torch.sum(output) / torch.sum(mask)
