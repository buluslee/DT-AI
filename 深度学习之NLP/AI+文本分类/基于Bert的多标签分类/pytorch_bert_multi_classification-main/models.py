from transformers import BertModel
import torch.nn as nn


class BertForMultiLabelClassification(nn.Module):
    def __init__(self, args):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(out_dims, args.num_tags)

    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert(
            input_ids = token_ids,
            attention_mask = attention_masks,
            token_type_ids = token_type_ids,
        )
        seq_out = bert_outputs[1]
        seq_out = self.dropout(seq_out)
        seq_out = self.linear(seq_out)
        return seq_out