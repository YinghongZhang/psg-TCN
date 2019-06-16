## 项目结构介绍

本项目的主要功能是识别PSG数据中的睡眠障碍事件

项目的结构树如下：

```
.
├─ cnn_claaifier.ipynb                血氧通道，使用CNN分类
├─ ml_classfier.ipynb                 血氧通道，使用传统机器学习模型分类
├─ data                               数据
│  ├─ train                           数据集
│  ├─ cnn_dataset.py                  用于CNN的dataset
│  └─ ml_dataset.py                   用于传统机器学习模型的dataset
├─ models                             CNN模型
│  ├─ dense.py
│  ├─ google_net.py
│  ├─ nasnet.py
│  ├─ psgnet.py
│  ├─ resnet.py
│  ├─ tcn.py
│  └─ vgg16.py
└─ utils                              工具模块
   ├─ logger.py                       日志模块
   └─ ml_detector.py                  异常检测
```

### 如何运行

用```jupyter notebook```打开```ml_classfier, cnn_claaifier```任意一个可直接运行

#### 准备数据集

数据集放在```data/train```下（目前有一个模拟文件），一份完整的数据（一个病人）包括```XX_脉搏.txt, XX_事件.txt 两个文件```。



