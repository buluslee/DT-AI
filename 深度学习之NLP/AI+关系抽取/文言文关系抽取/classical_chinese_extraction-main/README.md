# classical_chinese_extraction
文言文信息抽取（实体识别+关系抽取）。实体识别和关系抽取使用的网络均为globalpointer。

# 依赖

使用pytorch，并需要以下依赖。

```python
pip install datasets
pip install transformers
pip install tensorboardX
pip install seqeval
```

# 实体识别

进入到pytorch_GlobalPointer_Ner，执行：

```python
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/guwen/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=6 \
--head_size=64 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=256 \
--lr=5e-5 \
--other_lr=5e-5 \
--train_batch_size=16 \
--train_epochs=10 \
--eval_steps=100 \
--eval_batch_size=16 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.3 \
--use_tensorboard="False" \
--use_efficient_globalpointer="True"
```

得到：

```python

precision:0.5376 recall:0.7349 micro_f1:0.6210
INFO:__main__:          precision    recall  f1-score   support

     BOO       0.00      0.00      0.00         0
     WAR       0.00      0.00      0.00         0
     JOB       0.55      0.70      0.62       439
     LOC       0.52      0.52      0.52       218
     ORG       0.05      0.75      0.09         4
     PER       0.57      0.82      0.67       750

micro-f1       0.54      0.73      0.62      1411

INFO:__main__:冬十月，天子拜太祖兖州牧。十二月，雍丘溃，超自杀。夷邈三族。邈诣袁术请救，为其众所杀，兖州平，遂东略陈地。
INFO:__main__:{'JOB': [['兖州牧', 9]], 'LOC': [['雍丘', 17], ['兖州', 43], ['陈地', 50]], 'PER': [['天子', 4], ['太祖', 7], ['超', 21], ['邈', 26], ['邈', 30], ['袁术', 32]]}
```

# 关系抽取

进入到pytorch_GlobalPointer_triple_extraaction，执行：

```python
python main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/guwen/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=25 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=256 \
--lr=5e-5 \
--other_lr=5e-5 \
--train_batch_size=32 \
--train_epochs=10 \
--eval_steps=100 \
--eval_batch_size=8 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.3 \
--use_tensorboard="False" \
--use_dev_num=1000
```

得到：

```python
precision=0.2100 recall=0.1145 f1_score=0.1482

文本： 冬十月，天子拜太祖兖州牧。十二月，雍丘溃，超自杀。夷邈三族。邈诣袁术请救，为其众所杀，兖州平，遂东略陈地。
主体： [['邈']]
客体： [['兖州牧']]
关系： [[('邈', '任职', '兖州牧')]]
====================================================================================================
```

# 联合结果

```python
python get_result.py

{'JOB': [['兖州牧', 9]], 'LOC': [['雍丘', 17], ['兖州', 43], ['陈地', 50]], 'PER': [['天子', 4], ['太祖', 7], ['超', 21], ['邈', 26], ['邈', 30], ['袁术', 32]]}
文本： 冬十月，天子拜太祖兖州牧。十二月，雍丘溃，超自杀。夷邈三族。邈诣袁术请救，为其众所杀，兖州平，遂东略陈地。
主体： [['邈']]
客体： [['兖州牧']]
关系： [[('邈', '任职', '兖州牧')]]
====================================================================================================
```

# 补充

- 实体识别的效果还可以，但关系抽取的效果不尽人意，因为给的数据集里面每一条文本只包含一种关系，实际上可能包含多种关系的。
- 上述是对文言文数据集的一种尝试，整体框架很容易迁移到其他数据集上，具体每个模块下都有说明（其他数据）。
