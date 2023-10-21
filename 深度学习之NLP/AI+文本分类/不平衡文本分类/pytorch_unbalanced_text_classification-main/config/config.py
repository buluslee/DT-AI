import os


class BilstmForClassificationConfig:
    data_dir = './data/THUCNews/'
    save_dir = './checkpoints/'
    save_metric = './result/result.txt'

    with open(os.path.join(data_dir, 'cnews.labels'), 'r') as fp:
        labels = fp.read().strip().split('\n')
    label2id = {}
    id2label = {}
    for k, v in enumerate(labels):
        label2id[v] = k
        id2label[k] = v

    with open(os.path.join(data_dir, 'cnews.vocab.txt'), 'r') as fp:
        vocab = fp.read().strip().split('\n')
    vocab_size = len(vocab)

    do_train = True
    do_eval = False
    do_test = True
    do_predict = True

    is_unbalanced_data = True
    sampling_strategy = "None"  # undersample | oversample | datasetsample
    loss = "weight_celoss"  # celoss | weight_celoss | focalloss

    max_seq_len = 256
    word_embedding_dimension = 300
    hidden_dim = 384
    epochs = 10
    weight_decay = 0.01
    lr = 2e-5
    adam_epsilon = 1e-8
    warmup_proporation = 0.1
    train_batch_size = 32
    eval_batch_size = 32
