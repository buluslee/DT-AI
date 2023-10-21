import torch.nn as nn
from bertBaseModel import BaseModel


class BertMrcModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 args,
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param use_type_embed: type embedding for the sentence
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(BertMrcModel, self).__init__(bert_dir, dropout_prob=args.dropout_prob)

        out_dims = self.bert_config.hidden_size
        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        out_dims = mid_linear_dims

        self.start_fc = nn.Linear(out_dims, args.num_tags) # 218是标签的数目
        self.end_fc = nn.Linear(out_dims, args.num_tags)

        # reduction = 'none'
        # self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.criterion = nn.CrossEntropyLoss()

        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]

        self._init_weights(init_blocks)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                start_ids=None,
                end_ids=None):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out = bert_outputs[0]

        seq_out = self.mid_linear(seq_out)

        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)
        # print('start_logits.shape:',start_logits.shape)
        if start_ids is not None and end_ids is not None:
            start_logits = start_logits.view(-1, start_logits.shape[-1])
            end_logits = end_logits.view(-1, end_logits.shape[-1])
        return start_logits, end_logits

    def loss(self, start_ids, end_ids, start_logits, end_logits, token_type_ids):
        # 去掉 text_a 和 padding 部分的标签，计算真实 loss

        active_loss = token_type_ids.view(-1) == 1

        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]

        active_start_labels = start_ids.view(-1)[active_loss]
        active_end_labels = end_ids.view(-1)[active_loss]
        # print(active_start_labels.shape)
        # print(active_end_labels.shape)
        start_loss = self.criterion(active_start_logits, active_start_labels)
        end_loss = self.criterion(active_end_logits, active_end_labels)
        # print(start_loss.shape)
        # print(end_loss.shape)
        loss_val = start_loss + end_loss
        # print(loss_val.shape)
        return loss_val
