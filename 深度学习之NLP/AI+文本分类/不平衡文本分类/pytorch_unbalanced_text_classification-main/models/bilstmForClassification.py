from torch import nn
import torch.nn.functional as F


class BilstmForSequenceClassification(nn.Module):
    """
    Bidirectional LSTM running over word embeddings.
    """

    def __init__(self, num_labels: int, vocab_size: int, word_embedding_dimension: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.2,
                 bidirectional: bool = True):
        nn.Module.__init__(self)
        self.config_keys = ['num_labels', 'vocab_size', 'word_embedding_dimension', 'hidden_dim', 'num_layers',
                            'dropout', 'bidirectional']
        self.embedding = nn.Embedding(vocab_size, word_embedding_dimension)

        self.embeddings_dimension = hidden_dim
        if bidirectional:
            self.embeddings_dimension *= 2

        self.encoder = nn.LSTM(word_embedding_dimension, hidden_dim, num_layers=num_layers, dropout=dropout,
                               bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(self.embeddings_dimension, num_labels)

    def forward(self, src, sentence_lengths):
        # src:[batchsize, max_seq_len]
        token_embeddings = self.embedding(src)

        packed = nn.utils.rnn.pack_padded_sequence(token_embeddings, sentence_lengths, batch_first=True,
                                                   enforce_sorted=False)
        packed = self.encoder(packed)
        # 这里有一个参数可以控制填充到max_seq_len，total_length
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)[
            0]  # [batchsize, 当前批次中句子的最大长度, hidden_dim*2]
        output = unpack.permute(0, 2, 1).contiguous()
        output = F.adaptive_max_pool1d(output, output_size=1).squeeze()
        output = self.fc(output)
        return output


if __name__ == '__main__':
    model = BilstmForSequenceClassification(
        num_labels=10,
        vocab_size=5000,
        word_embedding_dimension=300,
        hidden_dim=384
    )
    for name, param in model.named_parameters():
        print(name)
