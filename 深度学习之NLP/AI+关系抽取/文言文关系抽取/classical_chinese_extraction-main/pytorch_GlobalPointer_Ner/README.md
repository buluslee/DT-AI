# pytorch_GlobalPointer_Ner

延申：
- 一种基于TPLinker_plus的命名实体识别：https://github.com/taishan1994/pytorch_TPLinker_Plus_Ner
- 一种one vs rest方法进行命名实体识别：https://github.com/taishan1994/pytorch_OneVersusRest_Ner
- 一种级联Bert用于命名实体识别，解决标签过多问题：https://github.com/taishan1994/pytorch_Cascade_Bert_Ner
- 一种多头选择Bert用于命名实体识别：https://github.com/taishan1994/pytorch_Multi_Head_Selection_Ner
- 中文命名实体识别最新进展：https://github.com/taishan1994/awesome-chinese-ner
- 信息抽取三剑客：实体抽取、关系抽取、事件抽取：https://github.com/taishan1994/chinese_information_extraction
- 一种基于机器阅读理解的命名实体识别：https://github.com/taishan1994/BERT_MRC_NER_chinese
- W2NER：命名实体识别最新sota：https://github.com/taishan1994/W2NER_predict

****

基于pytorch的GlobalPointer进行中文命名实体识别。

模型分别来自于参考中的【1】【2】。这里还是按照之前命名实体识别的相关模板，具体模型的介绍及预备知识请移步参考里面的链接。复现方式：

- 1、raw_data下新建一个process.py将原始数据处理为mid_data下的数据。
- 2、根据参数运行main.py以进行训练、验证、测试和预测。

模型和数据下载地址：链接：https://pan.baidu.com/s/1Gh9UQESQmEXuzyyPUG_FgQ?pwd=1a6s  提取码：1a6s

# 依赖

```
pytorch==1.6.0
transformers==4.5.0
seqeval
```

# 运行

```python
!python main.py \
--bert_dir="model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/cner/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=8 \
--head_size=64 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \
--lr=5e-5 \
--other_lr=5e-5 \
--train_batch_size=32 \
--train_epochs=7 \
--eval_steps=50 \
--eval_batch_size=8 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.1 \
--use_tensorboard="True" \
--use_efficient_globalpointer="True"
```

### 结果

globalpoint2.py也是可以用的，要选择它需要将main.py导入修改为```import globalpoint2```，并在使用模型时改为```globalpoint2.GlobalPointerNer```，模型名字自己设置为bert-2，参数use_efficient_globalpointer没有作用，因为是针对globalpoint.py的。

```python
precision:0.9559 recall:0.9565 micro_f1:0.9562
          precision    recall  f1-score   support

   TITLE       0.96      0.96      0.96       762
    RACE       1.00      0.93      0.97        15
    CONT       1.00      1.00      1.00        33
     ORG       0.94      0.94      0.94       539
    NAME       0.99      1.00      1.00       110
     EDU       0.97      0.99      0.98       109
     PRO       0.82      1.00      0.90        18
     LOC       1.00      1.00      1.00         2

micro-f1       0.96      0.96      0.96      1588

虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
Load ckpt from ./checkpoints/bert/model.pt
Use single gpu in: ['0']
{'TITLE': [['中共党员', 41], ['经济师', 50]], 'RACE': [['汉族', 18]], 'CONT': [['中国国籍', 21]], 'NAME': [['虞兔良', 1]], 'EDU': [['MBA', 46]], 'LOC': [['浙江绍兴人', 35]]}
```

默认使用的是globalpoint.py里面的模型，包含globalpointer和efficient-globalpoint，通过修改use_efficient_globalpointer来指定选择的模型，结果如下：

```python
globalpointer:
precision:0.9528 recall:0.9534 micro_f1:0.9531
          precision    recall  f1-score   support

   TITLE       0.95      0.95      0.95       762
    RACE       1.00      0.93      0.97        15
    CONT       1.00      1.00      1.00        33
     ORG       0.94      0.94      0.94       539
    NAME       0.99      1.00      1.00       110
     EDU       0.96      0.98      0.97       109
     PRO       0.86      1.00      0.92        18
     LOC       1.00      1.00      1.00         2

micro-f1       0.95      0.95      0.95      1588

虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
{'TITLE': [['中共党员', 41], ['经济师', 50]], 'RACE': [['汉族', 18]], 'CONT': [['中国国籍', 21]], 'NAME': [['虞兔良', 1]], 'EDU': [['MBA', 46]], 'LOC': [['浙江绍兴人', 35]]}

efficient-globalpoint:
precision:0.9616 recall:0.9616 micro_f1:0.9616
          precision    recall  f1-score   support

   TITLE       0.97      0.96      0.97       762
    RACE       1.00      0.93      0.97        15
    CONT       1.00      1.00      1.00        33
     ORG       0.95      0.94      0.94       539
    NAME       0.99      1.00      1.00       110
     EDU       0.97      0.98      0.98       109
     PRO       0.90      1.00      0.95        18
     LOC       1.00      1.00      1.00         2

micro-f1       0.96      0.96      0.96      1588

虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
{'TITLE': [['中共党员', 41], ['经济师', 50]], 'RACE': [['汉族', 18]], 'CONT': [['中国国籍', 21]], 'NAME': [['虞兔良', 1]], 'EDU': [['MBA', 46]], 'LOC': [['浙江绍兴人', 35]]}
```

### 补充

如果效果不好，尝试调小一些学习率。

# 参考

>[1]https://github.com/gaohongkui/GlobalPointer_pytorch
>
>[2]https://github.com/Tongjilibo/bert4torch/
>
>[3][将“softmax+交叉熵”推广到多标签分类问题 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/7359)
>
>[4][Transformer升级之路：2、博采众长的旋转式位置编码 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/8265)
>
>[5][GlobalPointer：用统一的方式处理嵌套和非嵌套NER - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/8373)
>
>[6][Efficient GlobalPointer：少点参数，多点效果 - 科学空间|Scientific Spaces (kexue.fm)](https://kexue.fm/archives/8877)

