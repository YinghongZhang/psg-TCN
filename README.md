## 项目结构介绍

本项目的主要功能是识别PSG数据中的睡眠障碍事件

项目的结构树如下：

```
:.
├─data
│  ├─train
├─log
├─models
└─utils
```

### data目录

data目录是存放训练数据和测试数据，数据集预处理代码的文件

train存放训练数据集，目前有一份数据文件，包含血氧、脉搏和事件数据，作为测试

### models目录

CNN网络模型，模型均已经改为一维卷积神经网络，部分模型是参考现有的CNN改为一维卷积，如文件名所示

### utils目录

异常检测，日志模块

### 运行

主文件夹下分为 ml_classfier, cnn_claaifier 两个ipython文件，数据集放在data/train下（目前有一个模拟文件用于测试），用jupyter notebook打开后可以直接运行

