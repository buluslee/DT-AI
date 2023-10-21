# pytorch_unbalanced_text_classification
基于Pytorch做了一些样本不平衡数据的中文文本分类实验。使用的数据集是THUCNews的部分数据。基本数据集是10类，每类数据5000。在processor/下的get_unbalanced_data.py用于生成样本不平衡的数据集，运行之后的数据：<br>
```
['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
nums = [5000, 4000, 3000, 2000, 1000, 500, 400, 300, 200, 100]
```
相关数据和保存好的日志模型下载：<br>
链接：https://pan.baidu.com/s/1JrK9vJChF2DqgwDr_62nrw?pwd=DTAI  <br>
提取码：DTAI<br>

# 依赖
```
pytorch
sklearn
numpy
imblance
```

# 说明
1、基于原始数据进行分类实现；<br>
2、使用processor/get_unbalanced_data.py进行不平衡数据集生成；<br>
3、采用过采样方法生成平衡数据集，并进行实验；<br>
4、采用欠采样方法生成平衡数据集，并进行实验；<br>
5、采用训练时随机采样生成平衡数据，并进行实验；<br>
6、采用带权重的交叉熵损失以及focal_loss损失进行实验；<br>
7、配置文件在config下，修改不同的配置参数即可，主运行程序在main.py下，包含训练、验证、测试以及预测。<br>

# 结果
|  模型   | accuracy  |precison  |recall  |macro_f1  |
|  ----  | ----  | ----  | ----  | ----  |
| bilstm  | 0.8973 | 0.9014  |  0.8973 |   0.8925|
| bilstm_unbalanced  | 0.4460 | 0.4526  |  0.4460 |   0.3548|
| bilstm_unbalanced_oversample  | 0.8817 | 0.8879  |  0.8817  |  0.8808 |
| bilstm_unbalanced_undersample  | 0.4674 | 0.5948 |   0.4674 |   0.3893
| bilstm_unbalanced_datasetsample  | 0.8183 | 0.8255 |  0.8183 | 0.7951|
| bilstm_unbalanced_focalloss  | 0.5770 | 0.5644  |  0.5770|    0.5239|
| bilstm_unbalanced_weight_celoss  | 0.4810 | 0.5523  |  0.4810 | 0.3982|

```
====================================
data_name：bilstm
model_name：bilstm
              precision    recall  f1-score   support

          体育     0.9871    0.9920    0.9895      1000
          财经     0.9082    0.9790    0.9423      1000
          房产     0.7287    0.8650    0.7910      1000
          家居     0.9072    0.5180    0.6595      1000
          教育     0.9221    0.8170    0.8664      1000
          科技     0.8668    0.9500    0.9065      1000
          时尚     0.8897    0.9760    0.9309      1000
          时政     0.8903    0.9410    0.9149      1000
          游戏     0.9338    0.9740    0.9535      1000
          娱乐     0.9806    0.9610    0.9707      1000

    accuracy                         0.8973     10000
   macro avg     0.9014    0.8973    0.8925     10000
weighted avg     0.9014    0.8973    0.8925     10000
北京市现已配售政策性住房4.1万套11月28日，记者从市住保办获悉，从政策房配售开始到目前，本市共公开配售政策性住房4.1万套。自本月起，另有2万套政策性住房也将陆续公开摇号配售。据介绍，从去年7月18日朝阳区组织首批经济适用住房公开摇号配售以来，全市共组织公开摇号49次，约4.6万户家庭参加了公开摇号，共公开配售政策性住房 4.1万套，其中经济适用住房1.6万套，限价商品住房2.5万套。而仅在今年，全市就共配售了1.8万套政策性住房。自本月起，本市还将有通州工具厂、大兴旧宫等经济适用住房项目以及昌平朱辛庄、溪城家园和大兴康庄等限价商品住房项目，共约2万套政策性住房陆续公开摇号配售。我要评论
预测标签：['房产']
真实标签：房产
====================================
```
# 补充
针对于数据不平衡问题，还可以采用数据增强的方式，可参考：<br>
https://github.com/taishan1994/eda_for_chinese_text_classification

# 参考
> https://github.com/qingkongzhiqian/NER_loss_compare/
>
> https://github.com/ufoym/imbalanced-dataset-sampler/
>
