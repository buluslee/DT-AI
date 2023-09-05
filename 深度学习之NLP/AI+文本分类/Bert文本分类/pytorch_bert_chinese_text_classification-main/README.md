# 基于pytorch+bert的中文文本分类

本项目是基于pytorch+bert的中文文本分类。
## 环境配置
 - 系统 Windows 10
 - python 3.9
 - cuda 11.6
 - GPU GTX1650

## 使用依赖
```python
torch==1.6.0
transformers==4.5.1
```
## 相关说明
```
--logs：存放日志
--checkpoints：存放保存的模型
--data：存放数据
--utils：存放辅助函数
--bert_config.py：相关配置
--data_loader.py：制作数据为torch所需的格式
--models.py：存放模型代码
--main.py：主运行程序，包含训练、验证、测试、预测以及相关评价指标的计算
```

在hugging face上预先下载好预训练的bert模型，放在和该项目同级下的model_hub文件夹下。
best.pt 请在百度网盘下载，放在checkpoints/cnews/里面 
新闻文本分类数据也在该网盘下面，放在data的cnews下
链接：https://pan.baidu.com/s/1Hxtn0phD7bLZAmVXrajcGA?pwd=DTAI 
提取码：DTAI 

## 裁判文书分类

数据集地址：[裁判文书网NLP文本分类数据集 - Heywhale.com](https://www.heywhale.com/mw/dataset/625869115fe0ad0017c6a7f7/file)

#### 一般步骤

- 1、在data下新建一个存放该数据集的文件夹，这里是cpws，然后将数据放在该文件夹下。在该文件夹下新建一个process.py，主要是获取标签并存储在labels.txt中。
- 2、在data_loader.py里面新建一个类，类可参考CPWSDataset，主要是返回一个列表：[(文本，标签)]。
- 3、在main.py里面的datasets、train_files、test_files里面添加上属于该数据集的一些信息，最后运行main.py即可。
- 4、在运行main.py时，可通过指定--do_train、--do_test、--do_predict来选择训练、测试或预测。

#### 运行指令
bachsize看显卡来调
```python
python main.py --bert_dir="../model_hub/chinese-bert-wwm-ext/" --data_dir="./data/cpws/" --data_name="cpws" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=5 --seed=123 --gpu_ids="0" --max_seq_len=256 --lr=3e-5 --train_batch_size=16 --train_epochs=5 --eval_batch_size=16 --do_train --do_test --do_predict
```
#### 结果

这里运行了300步之后手动停止了。

```python
{'盗窃罪': 0, '交通肇事罪': 1, '诈骗罪': 2, '故意伤害罪': 3, '危险驾驶罪': 4}
========进行测试========
【test】 loss：22.040314 accuracy：0.9823 micro_f1：0.9823 macro_f1：0.9822
              precision    recall  f1-score   support

         盗窃罪       0.97      0.99      0.98      1998
       交通肇事罪       0.97      0.99      0.98      1996
         诈骗罪       0.99      0.98      0.99      1998
       故意伤害罪       0.99      0.99      0.99      1999
       危险驾驶罪       0.99      0.95      0.97      2000

    accuracy                           0.98      9991
   macro avg       0.98      0.98      0.98      9991
weighted avg       0.98      0.98      0.98      9991
公诉机关指控：1、2015年3月18日18时许，被告人余某窜至漳州市芗城区丹霞路欣隆盛世小区2期工地内，趁工作人员不注意盗走工地内的脚手架扣件70个（价值人民币252元）。2、2015年3月19日13时和17时，被告人余某分两次窜至漳州市芗城区丹霞路欣隆盛世小区2期工地内一楼房一层的中间配电室内，利用随身携带的铁钳盗走该配电室内的电缆线（共计574米，价值人民币4707元）。3、2015年3月21日7时30分许，被告人余某窜至漳州市芗城区丹霞路欣隆盛世小区2期工地内一楼房一层靠东边的配电室内，利用随身携带的铁钳要将该配电室内的电缆线（共156米，价值人民币1279元）盗走时被工地负责人洪某某发现，后被工地保安吴某某抓获并扭送公安机关。公诉机关认为被告人余某的行为已构成××，本案第三起盗窃系犯罪未遂，建议对被告人余某在××至一年六个月的幅度内处以刑罚，并处罚金。
预测标签： 盗窃罪
真实标签： 盗窃罪
==========================
```

# 新闻文本分类

使用的数据集是THUCNews，数据地址：<a href="https://github.com/gaussic/text-classification-cnn-rnn">THUCNews</a>

#### 一般步骤

- 1、在data下新建一个存放该数据集的文件夹，这里是cnews，然后将数据放在该文件夹下。在该文件夹下新建一个process.py，主要是获取标签并存储在labels.txt中。
- 2、在data_loader.py里面新建一个类，类可参考CNEWSDataset，主要是返回一个列表：[(文本，标签)]。
- 3、在main.py里面的datasets、train_files、test_files里面添加上属于该数据集的一些信息，最后运行main.py即可。
- 4、在运行main.py时，可通过指定--do_train、--do_test、--do_predict来选择训练、测试或预测。

#### 运行
bachsize看显卡来调
```python
python main.py --bert_dir="hfl/chinese-bert-wwm-ext" --data_dir="./data/cnews/" --data_name="cnews" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=10 --seed=123 --gpu_ids="0" --max_seq_len=512 --lr=3e-5 --train_batch_size=4 --train_epochs=5 --eval_batch_size=4 --do_train --do_predict
```
#### 结果

这里运行了800步手动停止了。

```python
{'房产': 0, '娱乐': 1, '教育': 2, '体育': 3, '家居': 4, '时政': 5, '财经': 6, '时尚': 7, '游戏': 8, '科技': 9}
========进行测试========
【test】 loss：76.024950 accuracy：0.9697 micro_f1：0.9697 macro_f1：0.9696
              precision    recall  f1-score   support

          房产       0.91      0.92      0.92      1000
          娱乐       0.99      0.99      0.99      1000
          教育       0.97      0.96      0.97      1000
          体育       1.00      1.00      1.00      1000
          家居       0.98      0.91      0.94      1000
          时政       0.98      0.94      0.96      1000
          财经       0.96      0.99      0.97      1000
          时尚       0.94      1.00      0.97      1000
          游戏       0.99      0.99      0.99      1000
          科技       0.98      0.99      0.99      1000

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000

鲍勃库西奖归谁属？ NCAA最强控卫是坎巴还是弗神新浪体育讯如今，本赛季的NCAA进入到了末段，各项奖项的评选结果也即将出炉，其中评选最佳控卫的鲍勃-库西奖就将在下周最终四强战时公布，鲍勃-库西奖是由奈史密斯篮球名人堂提供，旨在奖励年度最佳大学控卫。最终获奖的球员也即将在以下几名热门人选中产生。〈〈〈 NCAA疯狂三月专题主页上线，点击链接查看精彩内容吉梅尔-弗雷戴特，杨百翰大学“弗神”吉梅尔-弗雷戴特一直都备受关注，他不仅仅是一名射手，他会用“终结对手脚踝”一样的变向过掉面前的防守者，并且他可以用任意一支手完成得分，如果他被犯规了，可以提前把这两份划入他的帐下了，因为他是一名命中率高达90%的罚球手。弗雷戴特具有所有伟大控卫都具备的一点特质，他是一位赢家也是一位领导者。“他整个赛季至始至终的稳定领导着球队前进，这是无可比拟的。”杨百翰大学主教练戴夫-罗斯称赞道，“他的得分能力毋庸置疑，但是我认为他带领球队获胜的能力才是他最重要的控卫职责。我们在主场之外的比赛(客场或中立场)共取胜19场，他都表现的很棒。”弗雷戴特能否在NBA取得成功？当然，但是有很多专业人士比我们更有资格去做出这样的判断。“我喜爱他。”凯尔特人主教练多克-里弗斯说道，“他很棒，我看过ESPN的片段剪辑，从剪辑来看，他是个超级巨星，我认为他很成为一名优秀的NBA球员。”诺兰-史密斯，杜克大学当赛季初，球队宣布大一天才控卫凯瑞-厄尔文因脚趾的伤病缺席赛季大部分比赛后，诺兰-史密斯便开始接管球权，他在进攻端上足发条，在ACC联盟(杜克大学所在分区)的得分榜上名列前茅，但同时他在分区助攻榜上也占据头名，这在众强林立的ACC联盟前无古人。“我不认为全美有其他的球员能在凯瑞-厄尔文受伤后，如此好的接管球队，并且之前毫无准备。”杜克主教练迈克-沙舍夫斯基赞扬道，“他会将比赛带入自己的节奏，得分，组织，领导球队，无所不能。而且他现在是攻防俱佳，对持球人的防守很有提高。总之他拥有了辉煌的赛季。”坎巴-沃克，康涅狄格大学坎巴-沃克带领康涅狄格在赛季初的毛伊岛邀请赛一路力克密歇根州大和肯塔基等队夺冠，他场均30分4助攻得到最佳球员。在大东赛区锦标赛和全国锦标赛中，他场均27.1分，6.1个篮板，5.1次助攻，依旧如此给力。他以疯狂的表现开始这个赛季，也将以疯狂的表现结束这个赛季。“我们在全国锦标赛中前进着，并且之前曾经5天连赢5场，赢得了大东赛区锦标赛的冠军，这些都归功于坎巴-沃克。”康涅狄格大学主教练吉姆-卡洪称赞道，“他是一名纯正的控卫而且能为我们得分，他有过单场42分，有过单场17助攻，也有过单场15篮板。这些都是一名6英尺175镑的球员所完成的啊！我们有很多好球员，但他才是最好的领导者，为球队所做的贡献也是最大。”乔丹-泰勒，威斯康辛大学全美没有一个持球者能像乔丹-泰勒一样很少失误，他4.26的助攻失误在全美遥遥领先，在大十赛区的比赛中，他平均35.8分钟才会有一次失误。他还是名很出色的得分手，全场砍下39分击败印第安纳大学的比赛就是最好的证明，其中下半场他曾经连拿18分。“那个夜晚他证明自己值得首轮顺位。”当时的见证者印第安纳大学主教练汤姆-克雷恩说道。“对一名控卫的所有要求不过是领导球队、使球队变的更好、带领球队成功，乔丹-泰勒全做到了。”威斯康辛教练博-莱恩说道。诺里斯-科尔，克利夫兰州大诺里斯-科尔的草根传奇正在上演，默默无闻的他被克利夫兰州大招募后便开始刻苦地训练，去年夏天他曾加练上千次跳投，来提高这个可能的弱点。他在本赛季与杨斯顿州大的比赛中得到40分20篮板和9次助攻，在他之前，过去15年只有一位球员曾经在NCAA一级联盟做到过40+20，他的名字是布雷克-格里芬。“他可以很轻松地防下对方王牌。”克利夫兰州大主教练加里-沃特斯如此称赞自己的弟子，“同时他还能得分，并为球队助攻，他几乎能做到一个成功的团队所有需要的事。”这其中四名球员都带领自己的球队进入到了甜蜜16强，虽然有3个球员和他们各自的球队被挡在8强的大门之外，但是他们已经表现的足够出色，不远的将来他们很可能出现在一所你熟悉的NBA球馆里。(clay)
预测标签： 体育
真实标签： 体育
==========================
```

# Dataparallel分布式训练

```python
python main_dataparallel.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/cnews/" \
--data_name="cnews" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=10 \
--seed=123 \
--gpu_ids="0,1,3" \
--max_seq_len=512 \
--lr=3e-5 \
--train_batch_size=64 \
--train_epochs=1 \
--eval_batch_size=64 \
--do_train \
--do_predict \
--do_test
```

# Distributed单机多卡分布式训练（windows下）

linux下没有测试过。运行需要在powershell里面运行，右键点击开始菜单，选择powershell。nvidia.bat用于监控运行之后GPU的使用情况。需要pytorch版本至少大于1.7，这里使用的是pytorch==1.12。

### 使用torch.distributed.launch启动

```python
python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=4 main_distributed.py --local_world_size=4 --bert_dir="../model_hub/chinese-bert-wwm-ext/" --data_dir="./data/cnews/" --data_name="cnews" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=10 --seed=123 --max_seq_len=512 --lr=3e-5 --train_batch_size=64 --train_epochs=1 --eval_batch_size=64 --do_train --do_predict --do_test
```

**说明**：文件里面通过```os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,3'```来选择使用的GPU。nproc_per_node为使用的GPU的数目，local_world_size为使用的GPU的数目。

### 使用torch.multiprocessing启动

```python
python main_mp_distributed.py --local_world_size=4 --bert_dir="../model_hub/chinese-bert-wwm-ext/" --data_dir="./data/cnews/" --data_name="cnews" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=10 --seed=123 --max_seq_len=512 --lr=3e-5 --train_batch_size=64 --train_epochs=1 --eval_batch_size=64 --do_train --do_predict --do_test
```

**说明**：文件里面通过```os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,3'```来选择使用的GPU。local_world_size为使用的GPU的数目。

# 补充

Q：怎么训练自己的数据集？<br>

A：按照样例的一般步骤里面进行即可。<br>

# 更新日志

- 2022-08-08：重构了代码，使得总体结构更加简单，更易于用于不同的数据集上。
- 2022-08-09：新增是否加载模型继续训练，运行参数加上--retrain。

- 2023-03-30：新增基于dataparallel的分布式训练。

- 2023-04-02：新增基于distributed的分布式训练。

