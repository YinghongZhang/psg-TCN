import os
import torch as t
from torch.utils import data
import numpy as np
import pandas as pd
from torchvision import transforms as T
from utils.ml_detector import EventDetectorForML
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis
import random
import matplotlib.pyplot as plot
from PIL import Image
import uuid


class BloodOxyen(data.Dataset):

    def __init__(self, root, train=True, extend=False):
        """
        Goal: get fragments, and divide data set
        Dataset of blood oxygen
        """
        self.train = train
        self.extend = extend

        detector = EventDetectorForML(root, extend)
        detector.detect()

        self.class_total = detector.class_total
        frag_num = len(detector.fragments)

        if self.train:
            self.frags = detector.fragments[:int(0.7 * frag_num)]
        else:
            self.frags = detector.fragments[int(0.7 * frag_num):]
        if extend:
            self.upsampling()

    def upsampling(self):
        """
        Data enhance
        """
        class_count = [0, 0, 0]
        for frag in self.frags:
            class_count[frag[1]] += 1
        class_max = max(class_count)
        print(class_count, class_max)

        for i in range(len(self.frags)):
            duplicate = class_max / class_count[self.frags[i][1]] - 1
            duplicate_integer = int(duplicate)
            duplicate_decimal = duplicate - int(duplicate)
            # print(self.frags[i][1], duplicate_integer, duplicate_decimal)

            self.frags.extend([self.frags[i]
                               for x in range(duplicate_integer)])
            if random.uniform(0, 1) < duplicate_decimal:
                self.frags.append(self.frags[i])

        class_count = [0, 0, 0]
        for frag in self.frags:
            class_count[frag[1]] += 1
        print(class_count)

    def __getitem__(self, index):
        """
        get one item from dataset
        """
        data = self.frags[index][0]
        label = self.frags[index][1]

        try:
            # features = t.from_numpy(self.extract_features(one_dataset[1]))
            # features = features.float()
            features = self.expand_values(data)
        except Exception as e:
            print(features, index)
        # print(features.size())
        # features = t.from_numpy(self.extract_features(one_dataset[1]))

        # print(type(features))

        return features, label
        # one_dataset = self.data[index]
        # img = self.get_img(one_dataset[1])
        # img = self.transforms(img)
        # return img, label

    def expand_values(self, features):
        """
        Train data dimension is [X, X, X]
        expand dimension of data 
        """
        # 拓展数据集的维度
        # expande dimension for dataset
        features = np.array(features)
        #features = features / np.sqrt(sum(features*features))
        #features = features - min(features) + 1
        features = t.from_numpy(features).float()
        features = t.unsqueeze(features, 0)
        return features

    def __len__(self):
        return len(self.frags)
