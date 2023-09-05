# predict.py

import torch

from model import efficientnetv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json


def main(img):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    # img = Image.open(r"C:\Users\qianc\Desktop\学习\vgg\flower\val\dandelion\10919961_0af657c4e8.jpg")  # 验证太阳花
    # # # img = Image.open("./roses.jpg")     #验证玫瑰花
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)  # 扩充维度，加bench维度

    # read class_indict
    try:
        json_file = open('flower\class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    net = efficientnetv2(num_classes=5)
    # load model weights
    model_weight_path = "MobileNet_v3\mobilenet_v3_small.pth"
    net.load_state_dict(torch.load(model_weight_path))  # 载入网络模型
    net.eval()  # 关闭dropout方法

    # 跟踪准确梯度
    with torch.no_grad():

        # predict class
        output = torch.squeeze(net(img))  # 压缩掉beach维度
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # 获取最大索引
    return class_indict[str(predict_cla)], predict[predict_cla].item()
    # plt.show()

