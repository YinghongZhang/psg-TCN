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
├─ checkpoints                        存放训练好的模型
└─ utils                              工具模块
   ├─ logger.py                       日志模块
   └─ ml_detector.py                  异常检测
```

### 如何运行

用```jupyter notebook```打开```ml_classfier, cnn_claaifier```任意一个可直接运行

#### 准备数据集

本项目使用的 PSG 数据位于```data/train```目录下，我们放置了一份数据包放在百度云盘里，链接为：[https://pan.baidu.com/s/1NUL4I8rKDFTxBjC-uE82nQ](https://pan.baidu.com/s/1NUL4I8rKDFTxBjC-uE82nQ%E3%80%82%E5%87%BA%E4%BA%8E%E4%BF%9D%E5%AF%86%E5%8E%9F%E5%9B%A0%EF%BC%8C%E5%AF%86%E7%A0%81%E8%AF%B7%E8%81%94%E7%B3%BB%E7%AE%A1%E7%90%86%E5%91%98%E8%8E%B7%E5%8F%96%E3%80%82) 出于保密原因，密码请联系管理员获取。

一份完整的数据（一个病人）包括XX_脉搏.txt, XX_事件.txt 两个文件。



