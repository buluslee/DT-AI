import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time


#device : GPU 或 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#数据预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), # 随机裁剪为224x224
                                 transforms.RandomHorizontalFlip(), # 水平翻转
                                 transforms.ToTensor(), # 转为张量
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),# 均值和方差为0.5
    "val": transforms.Compose([transforms.Resize((224, 224)), # 重置大小
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


batch_size = 32 # 批次大小
data_root = os.getcwd() # 获取当前路径
image_path = data_root + "/flower_data/"  # 数据路径

train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"]) # 加载训练数据集并预处理
train_num = len(train_dataset) # 训练数据集大小

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0) # 训练加载器

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"]) # 验证数据集
val_num = len(validate_dataset) # 验证数据集大小
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0) # 验证加载器

print("训练数据集大小: ",train_num,"\n") # 3306
print("验证数据集大小: ",val_num,"\n") # 364



net = AlexNet(num_classes=5, init_weights=True) # 调用模型

net.to(device)

loss_function = nn.CrossEntropyLoss() # 损失函数:交叉熵
optimizer = optim.Adam(net.parameters(), lr=0.0002) #优化器 Adam
save_path = './AlexNet.pth' # 训练参数保存路径
best_acc = 0.0 # 训练过程中最高准确率

#开始进行训练和测试，训练一轮，测试一轮
for epoch in range(10):
    # 训练部分
    print(">>开始训练: ",epoch+1)
    net.train()    #训练dropout
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad() # 梯度置0
        outputs = net(images.to(device)) 
        loss = loss_function(outputs, labels.to(device))
        loss.backward() # 反向传播
        optimizer.step()

        
        running_loss += loss.item() # 累加损失
        rate = (step + 1) / len(train_loader) # 训练进度
        a = "*" * int(rate * 50) # *数
        b = "." * int((1 - rate) * 50) # .数
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1) # 一个epoch花费的时间

    # 验证部分
    print(">>开始验证： ",epoch+1)
    net.eval()    #验证不需要dropout
    acc = 0.0  # 一个批次中分类正确个数
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            #print("outputs: \n",outputs,"\n")

            predict_y = torch.max(outputs, dim=1)[1]
            #print("predict_y: \n",predict_y,"\n")

            acc += (predict_y == val_labels.to(device)).sum().item() # 预测和标签一致，累加
        val_accurate = acc / val_num # 一个批次的准确率
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path) # 更新准确率最高的网络参数
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx 

#  {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将字典写入 json 文件
json_str = json.dumps(cla_dict, indent=4) # 字典转json
with open('class_indices.json', 'w') as json_file: # 对class_indices.json写入操作
    json_file.write(json_str) # 写入class_indices.json

