# fasttext_chinese_ABSA
基于fasttext的中文细粒度情感分类。<br>
由于协议这里不提供数据集，如需要则回复即可。<br>

数据集
链接：https://pan.baidu.com/s/1zfdiBxui1-mTKoYhxXDIyw?pwd=DTAI 
提取码：DTAI

# 相关依赖
```
pybind11>=2.2
Cython==0.28.5
future==0.16.0
jieba==0.39
numpy==1.15.1
pandas==0.23.4
python-dateutil==2.7.3
pytz==2018.5
scikit-learn==0.20rc1
scipy==1.1.0
six==1.11.0
#skift==0.0.21
#fasttext==0.9.2
```

# 执行步骤
1、修改config.py里面相关路径；<br>
2、安装相关依赖：```pip install -r requirements```。这里需要说明的是两个库：```skift```，该库是fasttext的类似于sklearn的调用方式，另一个就是```fasttext```，该项目是在linux下进行的，使用```pip install fasttext```的时候会报错，这里提供的解决方法是直接安装.whl文件，地址：https://pypi.org/project/fasttext-wheel/#files 。如果是windows环境，则可以去这里下载：https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext 。这样就可以避免c++环境带来的问题。虽然我们安装好了fasttext，但是在安装skift的时候还是会存在相同的问题，这里的解决方法是去skift的仓库，直接下载下来相关源码，并手动创建文件再引用，即```skift_core.py skift_util.py```。<br>
2、运行```python main_train.py```进行训练；<br>
3、运行```python main_test.py```进行测试；<br>
4、运行```python main_predict.py```进行预测并写入结果到相应文件中；<br>


# 参考
该项目基于：https://github.com/panyang/fastText-for-AI-Challenger-Sentiment-Analysis ，做了以下补充及修改：
1、解决```skift```和```fasttext```的安装问题；<br>
2、新增了```main_test.py```以报告的形式打印结果，而不仅仅是通过macro_f1；<br>

