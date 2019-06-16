import os
import torch as t
from torch.utils import data
import numpy as np
import pandas as pd
from torchvision import transforms as T
from utils.ml_detector import EventDetectorForML
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis
import matplotlib.pyplot as plot
from PIL import Image
import uuid


class BODataSet():

    def __init__(self, root, extend=False):
        """
        Goal: get fragments, and divide data set
        Data preprocess of blood oxygen for traditional mechining learning
        """
        if root is None:
            self.data_dir = '../data/train/'
        else:
            self.data_dir = root

        self.extend = extend

        detector = EventDetectorForML(root, extend)
        detector.detect()

        self.class_total = detector.class_total
        self.frag_num = len(detector.fragments)
        self.frags = detector.fragments
        # self.upsampling()

    def get_data(self):
        '''
        Get train data
        '''
        X = []
        Y = []

        for index in range(len(self.frags)):
            data = self.frags[index][0]
            label = self.frags[index][1]
            try:
                X.append(self.get_features(data))
                Y.append(label)
            except Exception as e:
                print(index)
            # print(features.size())
        return X, Y

    def get_features(self, values):
        '''
        Get different features and combine them
        '''
        features_array = []

        zero_order_features = self.get_zero_order_features(values)
        diff_features = self.get_diff_features(values)

        features_array.extend(zero_order_features)
        features_array.extend(diff_features)
        # features_array.extend(self.get_frequence_features(values))
        # feature_array.extend(values)
        # TODO: 加入更多频谱图的特征
        return np.array(features_array)

    def get_diff_features(self, values):
        '''
        statistics features of differential data
        max/min value, mean value
        '''
        # 异常检测可以用diff函数改写，就不用自己判断了
        # 差分结果的特征
        v = pd.DataFrame(values)
        v_diff = v.diff().fillna(0)
        v_big_0 = (v > 0)*v

        diff_mean = v_diff.mean()
        diff_var = v_diff.var()
        diff_max_v = v_diff.max()
        diff_min_v = v_diff.min()
        diff_skew = v_diff.skew()
        diff_kurt = v_diff.kurt()

        diff_range = diff_max_v - diff_min_v

        diffargmax = v_diff.idxmax()
        diffargmin = v_diff.idxmin()

        mean_big_0 = v_big_0.mean()
        var_big_0 = v_big_0.var()
        max_big_0 = v_big_0.max()
        min_big_0 = v_big_0.min()
        skew_big_0 = v_big_0.skew()
        kurt_big_0 = v_big_0.kurt()
        diff_range_big_0 = max_big_0 - min_big_0

        return [diff_mean, diff_var, diff_min_v, diff_skew, diff_kurt, diff_range, diffargmax, diffargmin, mean_big_0, var_big_0, max_big_0, min_big_0, skew_big_0, kurt_big_0]

    def get_zero_order_features(self, values):
        '''
        statistics features of data
        max/min value, mean value
        '''
        zero = values.count(0) > 0
        max_v, min_v = max(values), min(values)
        max_i, min_i = values.index(max_v), values.index(min_v)

        v_range = max_v - min_v

        mean = np.mean(values)
        var = np.var(values)
        std = np.std(values)

        med = np.median(values)
        avg_m_std = std / mean
        max_m_avg = max_v / mean
        min_m_avg = min_v / mean

        # 偏度和峰值
        s = pd.Series(values)
        skew = s.skew()
        kurt = s.kurt()

        return [max_v, min_v, mean, var, std, med, avg_m_std, max_m_avg, min_m_avg, skew, kurt]

    def get_frequence_features(self, values):
        '''
        get Spectrogram features
        '''
        Fs = 3  # 采样频率
        T_interval = 1 / Fs
        N = 10
        T = N/Fs
        k = np.arange(N)

        fft_res = np.fft.fft(values) / len(values)
        freqs = Fs*(k / T)

        amp = 2*np.abs(fft_res)
        phase = np.rad2deg(np.angle(fft_res))

        return [np.mean(amp), np.mean(freqs), np.mean(phase), amp.max(), phase.max(), freqs.max()]

    def upsampling(self):
        '''
        data enhance
        downsampling
        '''
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
