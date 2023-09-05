**1. 迁移虚拟环境**

 - 首先进入`C:\Users\<用户名>\.conda`文件夹，里面有3个文件，进入`envs`文件夹，复制里面的文件夹，然后找到你安装Anaconda的盘，比如`D:\Anaconda`，然后进入找到`D:\Anaconda\envs`文件夹。然后把你复制的文件夹粘贴进去，
 - 然后看看`D:\Anaconda`文件夹有没有`pkgs`包，没有就去`C盘.conda`文件复制`pkgs`包过去，如果Anaconda有`pkgs`包就去`C盘.conda`文件夹进入到`pkgs`，然后复制所有的文件，然后粘贴在`D:\Anaconda`的`pkgs`包，全部替换。
 - 注:`.conda`的txt文件(`environment.txt`)也复制到`D:\Anaconda`目录下，并打开`environments`把里面的路径换成`D:\Anaconda`中的虚拟环境路径，比如`D:\Anaconda\envs\pytorch_gpu`，同时`C:\`盘下的`environments`中路径也需要更改为`D:\Anaconda\envs\pytorch_gpu`(二个txt文件一致)。

**2. 设置环境变量**

右键此电脑-属性-高级系统设置-环境变量-系统变量，选中系统变量里的`Path`，点编辑，检查里面的环境变量，如果有`.conda`路径下的环境就全部替换成Anaconda路径下的环境，然后保存，如果没有就添加Anaconda相关环境变量（如`D:\Anaconda`、`D:\Anaconda\Lib`、`D:\Anaconda\Scripts`）。

**3. 配置VSCode编译器**

在VSCode编译器右下角选择Python解释器，添加内核，路径就是你的Anaconda下的`envs`的其中一个环境的`python.exe`。（此步可以先跳过，不影响环境迁移，只是做一个测试）

**4. 检查Anaconda和Jupyter终端路径**

点击开始-所有应用-找到anaconda prompt和Jupyter，右键打开文件所在位置并查看属性中的目标，查看目标里面的路径是否在`D:\`盘，如果不是`D:\`盘则把路径改为`D:\`。（避免后期anaconda与jupyter出现找不到路径的情况）

**5. 删除冗余文件夹**

将`C:\Users\<用户名>\.conda`目录下的文件夹全部删除，留下一个`environments`txt文件。

**6. 配置.condarc文件**

在`C:\Users\<用户名>`中找到`.condarc`文件，如果有该文件则在最后面添加如下四行，并将其中的路径换为自己的路径：

```markdown
envs_dirs:
  - D:\Anaconda\envs
pkgs_dirs:
  - D:\Anaconda\pkgs
```

如果没有找到该文件，则在Anaconda终端(anaconda prompt)输入`conda config`则会生成`.condarc`，然后在`C:\Users\<用户名>`目录下找到`.condarc`并将如下代码复制进去：

```markdown
show_channel_urls: true
channels:
  - defaults

envs_dirs:
  - D:\Anaconda\envs
pkgs_dirs:
  - D:\Anaconda\pkgs
```

**7. 设置文件夹权限**

在`D:\Anaconda`中找到`envs`文件夹并右键属性-安全-Users-编辑-将Users的权限全部开启(记得选中Users)，然后点击应用，确定，同理`pkgs`包也是一样操作。

**8. 测试**

测试，先去Anaconda终端测试`conda info --envs`看看是不是环境到`D:\`盘了，然后再激活一下那个环境，看看能不能激活，再进行一个安装包测试，比如`pip install dill`，安装好之后再进行`pip install dill`看安装的包的路径是不是安装到`D:\Anaconda`下了，然后再去VSCode终端进行同样的装包操作测试一下是否安装的包路径为D盘。

#### 至此就已经完成环境迁移的总过程

**遇到问题：**
若Jupyter内核运行失败，打开anaconda prompt激活对应环境，然后执行`pip install --user ipykernel`，然后执行`python -m ipykernel install --user --name=env_name`。(env_name即你自己的环境名)
