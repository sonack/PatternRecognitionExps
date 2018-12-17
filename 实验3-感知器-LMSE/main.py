#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import random
import pdb


data = np.array(
    [
        # class 0
        [1, 1],
        [2, 2],
        [2, 0],

        # class 1
        [0, 0],
        [1, 0],
        [0, 1]
    ]
)

label = np.array([0, 0, 0, 1, 1, 1])

mnist_train_data_path = '/home/snk/Documents/Homeworks/PatternRecognization/实验/Exp3/TrainSamples.csv'
mnist_train_label_path = '/home/snk/Documents/Homeworks/PatternRecognization/实验/Exp3/TrainLabels.csv'
mnist_test_data_path = '/home/snk/Documents/Homeworks/PatternRecognization/实验/Exp3/TestSamples.csv'
mnist_test_label_path = '/home/snk/Documents/Homeworks/PatternRecognization/实验/Exp3/TestLabels.csv'


def read_data(data_path, dtype=np.float):
    return np.genfromtxt(data_path, delimiter=',', dtype=dtype)



mnist_train_data = read_data(mnist_train_data_path)
mnist_train_label = read_data(mnist_train_label_path, dtype=np.int8)

mnist_test_data = read_data(mnist_test_data_path)
mnist_test_label = read_data(mnist_test_label_path)


# label 0, 1
class Perceptron:
    def __init__(self, w=None):
        self.w = w
        
    def augment_and_standardize(self, data, label):
        data = np.insert(data, 0, 1, axis=1)
        for idx, l in enumerate(label):
            if l:
                data[idx] = -data[idx]
        return data
    
    def init_params(self, dim_data):
        if not self.w:
            self.w = np.random.randn(dim_data)
    
    def judge_converge(self, data):
        for x in data:
            if np.dot(self.w, x) <= 0:
                return False
        return True
    
    def train(self, data, label):
        data = self.augment_and_standardize(data, label)
        n_data, dim_data = data.shape
        self.init_params(dim_data)
        k = 0
        while True:
            # make mistake
            if np.dot(self.w, data[k]) <= 0:
                self.w = self.w + data[k]
            k = (k + 1) % n_data
            if self.judge_converge(data):
                break
    
    def test(self, test_data, label):
        test_data = np.insert(test_data, 0, 1, axis=1)
        n_data = len(label)
        predict = np.empty_like(label)
        for i, x in enumerate(test_data):
            if np.dot(self.w, x) > 0:
                predict[i] = 0
            else:
                predict[i] = 1
        # print (predict)
        correct_cnt = (predict == label).sum()
        print ('[%d/%d] acc=%.2f%%' % (correct_cnt, n_data, correct_cnt / n_data * 100))

# label 0, 1
class LMSE:
    def augment_and_standardize(self, data, label):
        data = np.insert(data, 0, 1, axis=1)
        for idx, l in enumerate(label):
            if l:
                data[idx] = -data[idx]
        return data

    def train(self, data, label):
        data = self.augment_and_standardize(data, label)
        n_data, dim_data = data.shape
        label = np.ones((n_data, 1))
        self.w = np.linalg.inv(data.T.dot(data)).dot(data.T).dot(label).reshape((dim_data,))
    
    def test(self, test_data, label):
        test_data = np.insert(test_data, 0, 1, axis=1)
        n_data = len(label)
        predict = np.empty_like(label)
        for i, x in enumerate(test_data):
            if np.dot(self.w, x) > 0:
                predict[i] = 0
            else:
                predict[i] = 1
        # print (predict)
        correct_cnt = (predict == label).sum()
        print ('[%d/%d] acc=%.2f%%' % (correct_cnt, n_data, correct_cnt / n_data * 100))


# label 0, 1, ... , c-1
class KeslerPerceptron:
    def __init__(self, c, lr = 1e-6, ws = None):
        if isinstance(ws, list):
            assert len(ws) == c, "Init ws not enough for %d classes" % c

        self.c = c
        self.lr = lr
        self.ws = ws
    
    def init_params(self, dim_data):
        for i in range(self.c):
            self.ws.append(np.random.randn(dim_data))

    def augment(self, data):
        data = np.insert(data, 0, 1, axis=1)
        return data
    

    def judge_converge(self, data, label):
        for x,i in zip(data, label):
            g_i = np.dot(self.ws[i], x)
            for j in range(self.c):
                if j != i and np.dot(self.ws[j], x) > g_i:
                    return False
        return True

    def train(self, data, label, max_epoch = 15):
        data = self.augment(data)
        n_data, dim_data = data.shape
        if self.ws == None:
            self.ws = []
            self.init_params(dim_data)
        k = 0
        while True:
            # print ('k=%d' % k)
            x = data[k]
            i = label[k]
            g_i = np.dot(self.ws[i], x)
            for j in range(self.c):
                if j != i and np.dot(self.ws[j], x) >= g_i:
                    self.ws[i] += self.lr * x
                    self.ws[j] -= self.lr * x
            k = (k+1) % n_data
            if k == 0:
                max_epoch -= 1
                print ('epochs_remain:', max_epoch)
                print ('train:')
                self.test(mnist_train_data, mnist_train_label)
                print ('test:')
                self.test(mnist_test_data, mnist_test_label)

            if not max_epoch or self.judge_converge(data, label):
                break
        
    def test(self, test_data, label):
            test_data = np.insert(test_data, 0, 1, axis=1)
            n_data = len(label)
            predict = np.empty_like(label)
            for idx, x in enumerate(test_data):
                max_v = -9999999
                max_i = -1
                for i in range(self.c):
                    g_i = np.dot(self.ws[i], x)
                    if g_i > max_v:
                        max_v = g_i
                        max_i = i
                predict[idx] = max_i

            correct_cnt = (predict == label).sum()
            print ('[%d/%d] acc=%.2f%%' % (correct_cnt, n_data, correct_cnt / n_data * 100))


class multiclass_lmse_ova:
    def __init__(self, c):
        self.c = c
        self.lmses = [LMSE() for i in range(c)]

    def construct_data(self, data, label, cls):
        label_01 = label.copy()
        label_01[label==cls], label_01[label!=cls] = 0, 1
        return data, label_01

    def train(self, data, label):
        for idx, lmse in enumerate(self.lmses):
            # print ('training lmse %d ... ' % idx)
            data, label_01 = self.construct_data(data, label, idx)
            lmse.train(data, label_01)

    def test(self, test_data, label):
        test_data = np.insert(test_data, 0, 1, axis=1)
        n_data = len(label)
        predict = np.empty_like(label)
        for idx, x in enumerate(test_data):
            max_v = -99999
            max_i = -1
            for i in range(self.c):
                g = np.dot(self.lmses[i].w, x)
                if g > max_v:
                    max_v = g
                    max_i = i
            predict[idx] = max_i
        correct_cnt = (predict == label).sum()
        print ('[%d/%d] acc=%.2f%%' % (correct_cnt, n_data, correct_cnt / n_data * 100))
        
        
        
if __name__ == '__main__':
    # perceptron = Perceptron()
    # perceptron.train(data, label)  
    # perceptron.test(data, label)

    lmse = LMSE()
    lmse.train(data, label)
    lmse.test(data, label)

    # kesler = KeslerPerceptron(2)
    # kesler.train(data, label)
    # kesler.test(data, label)
 
    # mult_lmse = multiclass_lmse_ova(10)
    # mult_lmse.train(mnist_train_data, mnist_train_label)
    # print ('train:')
    # mult_lmse.test(mnist_train_data, mnist_train_label)
    # print ('test:')
    # mult_lmse.test(mnist_test_data, mnist_test_label)


    # kesler = KeslerPerceptron(10)
    # kesler.train(mnist_train_data, mnist_train_label, 10)
