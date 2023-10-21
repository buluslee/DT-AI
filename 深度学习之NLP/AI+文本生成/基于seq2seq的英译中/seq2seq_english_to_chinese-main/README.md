# seq2seq_english_to_chinese
基于pytorch的英文翻译成中文

# 说明
在config.py中有相关的配置，可以通过设置进行训练、测试、评估和预测。运行：
```python
python main.py
```

# 评估和预测
```python
Corpus BLEU: 19.718745812119977
输入：text = 'how old are you?'
翻译后的中文结果为:你几岁?,score:-1.188377857208252
翻译后的中文结果为:你几岁？,score:-1.4670623540878296
翻译后的中文结果为:你怎么样？,score:-3.0076115131378174
翻译后的中文结果为:您几岁？,score:-3.912196397781372
翻译后的中文结果为:你怎么了？,score:-4.185540676116943
```


# 参考
> https://blog.csdn.net/zp563987805/article/details/104451932<br>