# genius_for_your_data
参考GENIUS代码，使用GENIUS文本生成模型训练自己的数据集。

原代码地址：https://github.com/beyondguo/genius

原论文地址：https://arxiv.org/abs/2211.10330v1

演示地址：https://huggingface.co/spaces/beyond/genius

# 依赖

```python
pip install transformers
pip install dataset
pip install nltk
pip install rouge_score
pip install rouge
```

# 说明

- 1、huggingface上下载genius-base-chinese模型放在model_hub/genius-base-chinese/下。

- 2、数据格式参考data下的train.json，里面格式为：

	```python
	[("文本", "标签")]
	```

	当然，我们所需要的只是文本，你也可以是任意的格式，只需要在prepare_genius_pretrain_data_chinese_mine.py里面定义自己的加载数据方法就行。

- 3、运行```python genius_utils_mine.py```可测试一条数据。

- 4、修改pre_training/prepare_genius_pretrain_data_chinese_mine.py里面为自己数据加载的方式，修改```__main__```下面相关代码，执行```python prepare_genius_pretrain_data_chinese_mine.py```得到[MASK]的数据并存储。

- 5、修改pre_training/genius_pretrain_chinese.py里面相关设置，主要设置如下：

	```python
	dataset_path = '../data/data_with_sketch'  # [MASK]数据存储地址
	N = 40133
	# N为数据总数，这里由于数据较少，我们选择全部数据
	tokenized_dataset = dataset_with_sketch.select(random.sample(range(40133),k=N)).map(preprocess_function, 												batched=True, 
	                                        	   remove_columns=dataset_with_sketch.column_names,
	                                         	   batch_size=64,num_proc=8)  
	batch_size = 32 
	training_args = Seq2SeqTrainingArguments(
	    output_dir=output_dir,
	    evaluation_strategy="steps",
	    eval_steps = 500,  # 主要根据情况修改这里。  
	    save_strategy = 'epoch',
	    save_total_limit = num_train_epochs,
	    fp16 = True,
	    learning_rate=5.6e-5,
	    per_device_train_batch_size=batch_size,
	    per_device_eval_batch_size=batch_size,
	    weight_decay=0.01,
	    num_train_epochs=num_train_epochs,
	    predict_with_generate=True,
	    logging_steps=logging_steps,
	)
	
	# 选择1000条进行验证
	val_dataset = tokenized_dataset.select(range(1000))
	```

	最后运行```python genius_pretrain_chinese.py```即可。

- 5、最后根据保存的模型进行预测：

	```python
	# sega-chinese
	from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
	# checkpoint = '../model_hub//genius-base-chinese'
	checkpoint = '../saved_models/genius-base-chinese-data_with_sketch-40133/checkpoint-3765/'
	tokenizer = BertTokenizer.from_pretrained(checkpoint)
	sega_model = BartForConditionalGeneration.from_pretrained(checkpoint)
	sega_generator = Text2TextGenerationPipeline(sega_model, tokenizer, device=0)
	sega_generator
	
	"""
	银色的罗马高跟鞋，圆球吊饰耳饰单带，个性十足，有非常抢眼！
	稳吾到嘛？
	以后打死也不吃了
	来广州两天都没能织围脖,一直都在忙,再加上又感冒了,好痛苦[泪]不过广州给我的感觉灰常好!
	对骂我从来没怕过，你们也就只能考虑暗杀了，这样就充分保护动物了，臭傻逼们[打哈气]
	你这么说的我都不好意思呢
	我到了，文，好惨啊…
	"""
	
	sketchs = [
	  "银色的罗马高跟鞋，圆球吊饰耳饰单带，个性十足[MASK]抢眼[MASK]",
	  "稳吾到[MASK]",
	  "以后打死也不吃[MASK]",
	  "[MASK]广州两天都没能织围脖,一直[MASK]加上又感冒[MASK]痛苦[MASK]广州[MASK]感觉灰常好[MASK]",
	  "对骂我从来没怕[MASK]只能[MASK]暗杀[MASK]充分保护动物[MASK]逼们[MASK]哈气[MASK]",
	  "[MASK]这么[MASK]不好意思[MASK]",
	  "[MASK]好惨[MASK]",
	]
	for sketch in sketchs:
	    print('input sketch:\n>>> ', sketch)
	    print('SEGA-chinese output:\n>>> ',sega_generator(sketch, max_length=100, do_sample=True, num_beams=3)[0]['generated_text'].replace(' ',''),'\n')
	    
	input sketch:
	>>>  银色的罗马高跟鞋，圆球吊饰耳饰单带，个性十足[MASK]抢眼[MASK]
	SEGA-chinese output:
	>>>  银色的罗马高跟鞋，圆球吊饰耳饰单带，个性十足，很抢眼的一件装饰，很有女人味道，很喜欢，很好看，很实用，很时尚，很潮流。 
	
	input sketch:
	>>>  稳吾到[MASK]
	SEGA-chinese output:
	>>>  稳吾到家了！ 
	
	input sketch:
	>>>  以后打死也不吃[MASK]
	SEGA-chinese output:
	>>>  以后打死也不吃了！！！ 
	
	input sketch:
	>>>  [MASK]广州两天都没能织围脖,一直[MASK]加上又感冒[MASK]痛苦[MASK]广州[MASK]感觉灰常好[MASK]
	SEGA-chinese output:
	>>>  我在广州两天都没能织围脖,一直在忙,再加上又感冒又咳又痛苦,所以我只能去北京,去了广州就去了,感觉灰常好!!! 
	
	input sketch:
	>>>  对骂我从来没怕[MASK]只能[MASK]暗杀[MASK]充分保护动物[MASK]逼们[MASK]哈气[MASK]
	SEGA-chinese output:
	>>>  对骂我从来没怕过，只能说：我想暗杀那些没有充分保护动物的傻逼们，我也想打他们，可是我还是怕他们打我，给他们一个哈气。 
	
	input sketch:
	>>>  [MASK]这么[MASK]不好意思[MASK]
	SEGA-chinese output:
	>>>  我也这么说，可是我还是不好意思说。 
	
	input sketch:
	>>>  [MASK]好惨[MASK]
	SEGA-chinese output:
	>>>  我好惨啊！ 
	```

# 补充

怎么用于数据增强这里没有继续下去了，大体看了一下：

- 针对于命名实体识别而言，关键词是由结巴得到再加上实体得到。也就是这些词是不会被[MASK]掉的。然后可以通过模型生成相关的上下文。最后在重新计算实体的位置。
- 对于文本分类而言，方法可以有很多种。可以通过随机[MASK]，然后用预测文本替换[MASK]。也可以像上面一样先选出关键词，再生成关键词的上下文，最后组成文本。
- 以后看再补充数据增强的实例。

最后，感谢原作者的相关工作，感兴趣的可以去读读论文，上手试一下。

