import torch
import torch.nn as nn
from transformers import BertModel

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_true = y_true.view(y_true.shape[0]*y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
    y_pred = y_pred.view(y_pred.shape[0]*y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, heads, head_size, RoPE=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.dense = nn.Linear(hidden_size, self.heads * self.head_size * 2)
        self.RoPE = RoPE
    
    
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings
        
    def forward(self, inputs, attention_mask):

        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(inputs)
        outputs = torch.split(outputs, self.head_size * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[...,:self.head_size], outputs[...,self.head_size:] # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_size).to(outputs.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
            
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1) 
        logits = logits - mask * 1e12
        
        return logits/self.head_size**0.5

  
class GlobalPointerNer(nn.Module):
  def __init__(self, args):
      super().__init__()
      self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True,
                          hidden_dropout_prob=args.dropout_prob)
      self.global_pointer = GlobalPointer(hidden_size=768, heads=args.num_tags, head_size=args.head_size)

  def forward(self, token_ids, attention_masks, token_type_ids, labels=None):
      output = self.bert(token_ids, attention_masks, token_type_ids)  # [btz, seq_len, hdsz]
      sequence_output = output[0]
      logits = self.global_pointer(sequence_output, attention_masks.gt(0).long())
      if labels is None:
        # scale返回
        return logits
      
      loss = multilabel_categorical_crossentropy(logits, labels)
      return loss, logits