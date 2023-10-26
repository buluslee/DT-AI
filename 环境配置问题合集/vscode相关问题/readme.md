## 1.vscode中import torch后终端报错显示conda-script.py error:argument COMMAND
![d232275f6b3fdb5c897d160ff02e8c4](https://github.com/buluslee/DT-AI/assets/142234262/f69017a4-761d-4233-8e74-e20cf2e565a8)

## 解决办法
管理员身份打开windows powershell，然后输入Set-ExecutionPolicy -ExecutionPolicy RemoteSigned，然后输入y

然后管理员打开anaconda powershell prompt，输入conda init powershell，重启vscode重新运行

## 2.vscode中imoprt torch后终端报错No module name torch
<img width="1163" alt="55fd79095d29be9f5096b4ac627cf4b" src="https://github.com/buluslee/DT-AI/assets/142234262/958ea471-4bb8-4760-81aa-20434dc1547d">

## 解决办法
在vscode右下角选择已经装了pytroch的环境，若选择已装pytorch的环境后依然报此错误与问题一的解决方法一样

## 3.vscode中没有新建立的环境，无法找到对应的python解释器
<img width="416" alt="image" src="https://github.com/buluslee/DT-AI/assets/142234262/8567a227-abe0-42f2-9ec6-23efc9083497">

## 解决办法
请查看vscode安装及使用教程里面

## 4.vscode安装remote development插件显示不兼容
<img width="416" alt="image" src="https://github.com/buluslee/DT-AI/assets/142234262/4078e743-3106-4b60-8c72-9db84d6e4a1e">

## 解决办法
更新vscode到最新版本，官网重下最新版本覆盖或者设置里面找到更新

## 5.中文用户名在anaconda powershell prompt中输入conda init powershell后最后一行profile.psl文件显示到乱码文件里面
<img width="603" alt="5acce0ffc3a21f2dca5c1f63cee09a2" src="https://github.com/buluslee/DT-AI/assets/142234262/e46d11dc-fa1d-43b2-ad7e-23dce75a7995">

## 解决办法
进入C盘用户下的乱码文件夹、进入Documents\WindowsPowerShell找到里面的profile.psl文件并复制，将此profile.psl文件复制到正确用户名(即中文名)下的文件(Documents)下面的WindowsPowerShell文件夹里面，
若没有WindowsPowerShell文件夹则手动创建一个，完成之后重启vscode运行即可



