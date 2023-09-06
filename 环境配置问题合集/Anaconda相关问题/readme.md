## 1.nvidia-smi不是内部或外部命令
<img width="416" alt="image" src="https://github.com/buluslee/DT-AI/assets/142234262/cc91f102-ac6e-4742-9e85-59399633a846">

## 解决办法
显卡不是NVIDIA型的，可以到显示适配器检查下显卡类型，若没有则下载pytorch用cpu版本

若有NVIDIA显卡则检查显卡驱动是否正确安装。你可以通过设备管理器来进行检查。按下Windows键+X，选择"设备管理器"，然后在设备管理器中找到"显示适配器"并展开，看是否有NVIDIA显卡列出。如果出现感叹号，那可能意味着驱动程序存在问题。

确保你的NVIDIA驱动是最新的。可以通过[NVIDIA官网的驱动下载页面](https://www.nvidia.cn/Download/index.aspx)，输入你的显卡型号，然后下载并安装最新的驱动。

检查nvidia-smi的路径。在Windows的命令提示符（CMD）或Powershell中，输入"where nvidia-smi"命令。如果没有返回值或者返回"无法找到文件"，那可能意味着nvidia-smi没有添加到你的系统路径中。你需要找到nvidia-smi的实际位置（通常在"NVIDIA Corporation\NVSMI"文件夹内，例如"C:\Program Files\NVIDIA Corporation\NVSMI"或者是system32中），并将其添加到系统环境变量的Path中。

## 2.vscode中报错显示无法将conda识别
<img width="497" alt="image" src="https://github.com/buluslee/DT-AI/assets/142234262/24bb0abe-f835-47f2-8b5f-60e68990a8ea">

## 解决办法
右键此电脑-属性-高级系统设置-环境变量-系统变量-Path-编辑，然后添加你自己anaconda目录下的如下路径：

D:\Anaconda

D:\Anaconda\Scripts

D:\Anaconda\Library\bin

## 3.jupyter网页可以弹开，但是无法运行出结果
<img width="1057" alt="92638fa549672cfae11e0a5f8af3272" src="https://github.com/buluslee/DT-AI/assets/142234262/c1f8adcf-b4cb-43da-9ad1-4e878bbb8e32">

## 解决办法
通过降低pyzmq版本可以解决，打开anaconda prompt，进入相应的环境，输入如下代码

pip uninstall pyzmq

pip install pyzmq==19.0.2
