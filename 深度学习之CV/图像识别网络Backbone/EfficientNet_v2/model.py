import torch
from torch import nn
from functools import partial  # 传参
# from torchstat import stat  # 查看网络参数
 
# -------------------------------------- #
# Stochastic Depth dropout 方法，随机丢弃输出层
# -------------------------------------- #
 
def drop_path(x, drop_prob: float=0., training: bool=False):  # drop_prob代表丢弃概率
    # （1）测试时不做 drop path 方法
    if drop_prob == 0. or training is False:
        return x
    # （2）训练时使用
    keep_prob = 1 - drop_prob  # 网络每个特征层的输出结果的保留概率
    shape = (x.shape[0],) + (1,) * (x.ndim-1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output
 
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        out = drop_path(x, self.drop_prob, self.training)
        return out
 
class SequeezeExcite(nn.Module):
    def __init__(self,
                 input_c,  # 输入到MBConv模块的特征图的通道数
                 expand_c,  # 输入到SE模块的特征图的通道数
                 se_ratio=0.25,  # 第一个全连接下降的通道数的倍率
                ):
        super(SequeezeExcite, self).__init__()
 
        # 第一个全连接下降的通道数
        sequeeze_c = int(input_c * se_ratio)
        # 1*1卷积代替全连接下降通道数
        self.conv_reduce = nn.Conv2d(expand_c, sequeeze_c, kernel_size=1, stride=1)
        self.act1 = nn.SiLU()
        # 1*1卷积上升通道数
        self.conv_expand = nn.Conv2d(sequeeze_c, expand_c, kernel_size=1, stride=1)
        self.act2 = nn.Sigmoid()
    
    # 前向传播
    def forward(self, x):
        # 全局平均池化[b,c,h,w]==>[b,c,1,1]
        scale = x.mean((2,3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        # 对输入特征图x的每个通道加权
        return scale * x

class ConvBnAct(nn.Module):
    def __init__(self,
                 in_planes,  # 输入特征图通道数
                 out_planes,  # 输出特征图通道数
                 kernel_size=3,  # 卷积核大小
                 stride=1,  # 滑动窗口步长
                 groups=1,  # 卷积时通道数分组的个数
                 norm_layer=None,  # 标准化方法
                 activation_layer=None,  # 激活函数
                ):
        super(ConvBnAct, self).__init__()
 
        # 计算不同卷积核需要的0填充个数
        padding = (kernel_size - 1) // 2
        # 若不指定标准化和激活函数，就用默认的
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        
        # 卷积
        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False,
                              )
        # BN
        self.bn = norm_layer(out_planes)
        # silu
        self.act = activation_layer(inplace=True)
    
    # 前向传播
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
 
 
# -------------------------------------- #
# MBConv卷积块
# -------------------------------------- #
 
class MBConv(nn.Module):
    def __init__(self, 
                 input_c,
                 output_c,
                 kernel_size, # DW卷积的卷积核size
                 expand_ratio,  # 第一个1*1卷积上升通道数的倍率
                 stride,  # DW卷积的步长
                 se_ratio,  # SE模块的第一个全连接层下降通道数的倍率
                 drop_rate,  # 随机丢弃输出层的概率
                 norm_layer,
                 ):
        super(MBConv, self).__init__()
 
        # 下采样模块不存在残差边；基本模块stride=1时且输入==输出特征图通道数，才用到残差边
        self.has_shortcut = (stride==1 and input_c==output_c)
        # 激活函数
        activation_layer = nn.SiLU
        # 第一个1*1卷积上升的输出通道数
        expanded_c = input_c * expand_ratio
 
        # 1*1升维卷积
        self.expand_conv = ConvBnAct(in_planes=input_c,  # 输入通道数
                                     out_planes=expanded_c,  # 上升的通道数
                                     kernel_size=1,
                                     stride=1,
                                     groups=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer,
                                     )
        # DW卷积
        self.dwconv = ConvBnAct(in_planes=expanded_c,  
                                out_planes=expanded_c,  # DW卷积输入和输出通道数相同
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,  # 对特征图的每个通道做卷积
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                )
        # SE注意力
        # 如果se_ratio>0就使用SE模块，否则线性连接、
        if se_ratio > 0:
            self.se = SequeezeExcite(input_c=input_c,  # MBConv模块的输入通道数
                                    expand_c=expanded_c,  # SE模块的输出通道数
                                    se_ratio=se_ratio,  # 第一个全连接的通道数下降倍率
                                    )
        else:
            self.se = nn.Identity()
        
        # 1*1逐点卷积降维
        self.project_conv = ConvBnAct(in_planes=expanded_c,
                                      out_planes=output_c,
                                      kernel_size=1,
                                      stride=1,
                                      groups=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity,  # 不使用激活函数，恒等映射
                                      )
        # droppath方法
        self.drop_rate = drop_rate
        # 只在基本模块使用droppath丢弃输出层
        if self.has_shortcut and drop_rate>0:
            self.dropout = DropPath(drop_prob=drop_rate)
        
    # 前向传播
    def forward(self, x):
        out = self.expand_conv(x)  # 升维
        out = self.dwconv(out)  # DW
        out = self.se(out)  # 通道注意力
        out = self.project_conv(out)  # 降维
 
        # 残差边
        if self.has_shortcut:
            if self.drop_rate > 0:
                out = self.dropout(out)  # drop_path方法
            out += x  # 输入和输出短接
        return out

class FusedMBConv(nn.Module):
    def __init__(self,
                 input_c,
                 output_c,
                 kernel_size,  # DW卷积核的size
                 expand_ratio,  # 第一个1*1卷积上升的通道数
                 stride,  # DW卷积的步长
                 se_ratio,  # SE模块第一个全连接下降通道数的倍率
                 drop_rate,  # drop—path丢弃输出层的概率
                 norm_layer,
                 ):
        super(FusedMBConv, self).__init__()
 
        # 残差边只用于基本模块
        self.has_shortcut = (stride==1 and input_c==output_c)
        # 随机丢弃输出层的概率
        self.drop_rate = drop_rate
        # 第一个卷积是否上升通道数
        self.has_expansion = (expand_ratio != 1)
        # 激活函数
        activation_layer = nn.SiLU
 
        # 卷积上升的通道数
        expanded_c = input_c * expand_ratio
 
        # 只有expand_ratio>1时才使用升维卷积
        if self.has_expansion:
            self.expand_conv = ConvBnAct(in_planes=input_c,
                                         out_planes=expanded_c,  # 升维的输出通道数
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer,
                                         )
            # 1*1降维卷积
            self.project_conv = ConvBnAct(in_planes=expanded_c,
                                          out_planes=output_c,
                                          kernel_size=1,
                                          stride=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity,
                                          )
        # 如果expand_ratio=1，即第一个卷积不上升通道
        else:
            self.project_conv = ConvBnAct(in_planes=input_c,
                                          out_planes=output_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer,
                                          )
        
        # 只有在基本模块中才使用shortcut，只有存在shortcut时才能用drop_path
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate>0:
            self.dropout = DropPath(drop_rate)
 
    # 前向传播
    def forward(self, x):
        # 第一个卷积块上升通道数倍率>1
        if self.has_expansion:
            out = self.expand_conv(x)
            out = self.project_conv(out)
        # 不上升通道数
        else:
            out = self.project_conv(x)
 
        # 基本模块中使用残差边
        if self.has_shortcut:
            if self.drop_rate > 0:
                out = self.dropout(out)
            out += x
        return out
 
class EfficientNetV2(nn.Module):
    def __init__(self, 
                model_cnf:list,  # 配置文件
                num_classes=1000,  # 分类数
                num_features=1280, # 输出层的输入通道数
                drop_path_rate=0.2, # 用于卷积块中的drop_path层
                drop_rate=0.2):  # 输出层的dropout层
        super(EfficientNetV2, self).__init__()
 
        # 为BN层传递默认参数
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
 
        # 第一个卷积层的输出通道数
        stem_filter_num = model_cnf[0][4]  # 24
 
        # 第一个卷积层[b,3,h,w]==>[b,24,h//2,w//2]
        self.stem = ConvBnAct(in_planes=3,
                              out_planes=stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer,
                              )
        # 统计一共堆叠了多少个卷积块
        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []  # 保存所有的卷积块
 
        # 遍历每个stage的参数
        for cnf in model_cnf:
            # 当前stage重复次数
            repeats = cnf[0]
            # 使用何种卷积块，0标记则用FusedMBConv
            op = FusedMBConv if cnf[-2]==0 else MBConv
            # 堆叠每个stage
            for i in range(repeats):
                blocks.append(op(
                    input_c=cnf[4] if i==0 else cnf[5],  # 只有第一个下采样卷积块的输入通道数需要调整，其余都一样
                    output_c=cnf[5],  # 输出通道数保持一致
                    kernel_size=cnf[1],  # 卷积核size
                    expand_ratio=cnf[3],  # 第一个卷积升维倍率
                    stride=cnf[2] if i==0 else 1,  # 第一个卷积块可能是下采样stride=2，剩下的都是基本模块
                    se_ratio=cnf[-1],  # SE模块第一个全连接降维倍率
                    drop_rate=drop_path_rate * block_id / total_blocks,  # drop_path丢弃率满足线性关系
                    norm_layer=norm_layer,
                    ))
                # 没堆叠完一个就计数
                block_id += 1
            # 以非关键字参数形式返回堆叠后的stage
        self.blocks = nn.Sequential(*blocks)
 
        # 输出层的输入通道数 256
        head_input_c = model_cnf[-1][-3]
 
        # 输出层
        self.head = nn.Sequential(
            # 1*1卷积 [b,256,h,w]==>[b,1024,h,w]
            ConvBnAct(head_input_c, num_features, kernel_size=1, stride=1, norm_layer=norm_layer),
            # 全剧平均池化[b,1024,h,w]==>[b,1024,1,1]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), # [b,1024]
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(num_features, num_classes)  # [b,1000]
        )
 
        # ----------------------------------------- #
        # 权重初始化
        # ----------------------------------------- #     
 
        for m in self.modules():
            # 卷积层初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # BN层初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 全连接层
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    # 前向传播
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x       
 
def efficientnetv2(num_classes=1000):
    # 配置文件
    # repeat, kernel, stride, expansion, in_c, out_c, operate, squeeze_rate
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25],
                    ]
 
    # 构建模型
    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes)
 
    return model
 
# ----------------------------------------- #
# 查看网络参数量
# ----------------------------------------- #      
 
if __name__ == '__main__':
 
    model = efficientnetv2()
 
    # 构造输入层shape==[4,3,224,224]
    inputs = torch.rand(4,3,224,224)
    
    # 前向传播查看输出结果
    outputs = model(inputs)
    print(outputs.shape)  # [4,1000]
 
    # 查看模型参数，不需要指定batch维度
    # stat(model, input_size=[3,224,224])  
 
    '''
    Total params: 21,458,488
    Total memory: 144.98MB
    Total MAdd: 5.74GMAdd
    Total Flops: 2.87GFlops
    Total MemR+W: 267.99MB
    '''