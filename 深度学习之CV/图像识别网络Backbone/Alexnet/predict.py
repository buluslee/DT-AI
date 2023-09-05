import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json


# 加载图片
img = Image.open("./sunflower.jpg")  #验证太阳花

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#img = Image.open("./roses.jpg")     #验证玫瑰花
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# 读取 class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 调用模型
model = AlexNet(num_classes=5)
# 加载模型参数
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval() # 不用Dropout
with torch.no_grad():
    output = torch.squeeze(model(img)) # 压缩
    predict = torch.softmax(output, dim=0) # 生成概率
    predict_cla = torch.argmax(predict).numpy() # 最大值的索引
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()
