#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import random
import math
import pdb
import copy
np.seterr(divide='ignore',invalid='ignore')

GMM1_data = '/Users/snk/Downloads/实验/Exp2/Train1.csv'
GMM2_data = '/Users/snk/Downloads/实验/Exp2/Train2.csv'


GMM1_testdata = '/Users/snk/Downloads/实验/Exp2/Test1.csv'
GMM2_testdata = '/Users/snk/Downloads/实验/Exp2/Test2.csv'

MNIST_traindata = '/Users/snk/Downloads/实验/Exp2/TrainSamples.csv'
MNIST_trainlabel = '/Users/snk/Downloads/实验/Exp2/TrainLabels.csv'

MNIST_testdata = '/Users/snk/Downloads/实验/Exp2/TestSamples.csv'
MNIST_testlabel = '/Users/snk/Downloads/实验/Exp2/TestLabels.csv'

mixture_num = 2
data_dim = 17
class_num = 10

threshold = 1e-30
eps = 1e-100



all_alphas = []
all_mus = []
all_sigmas = []

def classify(xs, true_label):
    len_x = len(xs)
    correct_cnt = 0
    for t in range(len_x):
        probs = []
        for i in range(class_num):
            alphas = all_alphas[i]
            mus = all_mus[i]
            sigmas = all_sigmas[i]

            p = 0
            for i in range(mixture_num):
                p += alphas[i] * pdf_multivariate_gauss(xs[t], mus[i], sigmas[i])
            probs.append(p)

        max_v = probs[0]
        max_i = 0
        for i, prob in enumerate(probs[1:], 1):
            if prob > max_v:
                max_v = prob
                max_i = i
        if max_i + 1 == true_label:
            correct_cnt += 1

    print ('[%d/%d]' % (correct_cnt, len_x))

def random_init_params():
    print ('Init Params')
    global alphas, mus, sigmas
    alphas = [random.random() for i in range(mixture_num)]
    alphas_sum = sum(alphas)
    alphas = [alpha/alphas_sum for alpha in alphas]
    mus = [np.random.randn(data_dim, 1) for i in range(mixture_num)]
    sigmas = []
    for i in range(mixture_num):
        sigma = np.random.rand(data_dim, data_dim)
        sigma = sigma.dot(sigma.T)
        sigmas.append(sigma)

def pdf_multivariate_gauss(x, mu, cov):
    if abs(np.linalg.det(cov)) < eps:
        global is_singular
        is_singular = True
        return 0
    
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov + eps * np.eye(x.shape[0]))**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov + eps * np.eye(x.shape[0])))).dot((x-mu))
    return float(part1 * np.exp(part2))

def estimate_y():
    print ('estimate y')
    global ys
    for t in range(n_data):
        normalizer = 0
        for i in range(mixture_num):
            ys[t][i] = alphas[i] * pdf_multivariate_gauss(xs[t], mus[i], sigmas[i])
            normalizer += ys[t][i]
        # print (normalizer)
        for i in range(mixture_num):
            ys[t][i] /= (normalizer + eps)
        
def estimate_theta():
    print ('estimate theta')
    # estimate alpha
    for i in range(mixture_num):
        alphas[i] = 0
        for t in range(n_data):
            alphas[i] += ys[t][i]
        alphas[i] /= n_data 
    # estimate mu & sigma
    for i in range(mixture_num):
        mus[i].fill(0)
        sigmas[i].fill(0)

        normalizer = 0
        for t in range(n_data):
            mus[i] += ys[t][i] * xs[t]
            normalizer += ys[t][i]  
        mus[i] /= (normalizer + eps)

        for t in range(n_data):
            sigmas[i] += ys[t][i] * (xs[t] - mus[i]).dot((xs[t] - mus[i]).T)   
        sigmas[i] /= (normalizer + eps)
    

def judge_converge(last_likelihood, likelihood):
    # print (abs(last_likelihood - likelihood))
    return abs(last_likelihood - likelihood) < threshold

# log likelihood in fact
def calc_likelihood():
    global is_overflow
    ll = 0
    for t in range(n_data):
        p = 0
        for i in range(mixture_num):
            p += alphas[i] * pdf_multivariate_gauss(xs[t], mus[i], sigmas[i])
        if abs(p) < eps:
            is_overflow = True
            return ll
        ll += math.log(p)

    return ll


def main():

    global is_singular, is_overflow
    last_likelihood = 1
    iteration = 0

    random_init_params()

    while True:
        estimate_y()
        estimate_theta()
        likelihood = calc_likelihood()
        if is_singular or is_overflow:
            print ('sigular:%d overflow:%d' % (is_singular, is_overflow))
            is_singular = False
            is_overflow = False
            print ('ReRun')
            main()
            return
        # print ('ll:', likelihood)
        if judge_converge(last_likelihood, likelihood):
            break
        last_likelihood = likelihood
        iteration += 1
        print ("Iter : %d" % iteration)

    print ('over')
    print ('estimated params:')
    print ('-'*10+' alphas '+'-'*10)
    for i in range(mixture_num):
        print ('alpha[%d]=%f' % (i, alphas[i]))

    print ('-'*10+' mus '+'-'*10)
    for i in range(mixture_num):
        print ('mu[%d]=\n\t%s' % (i, mus[i]))
    
    print ('-'*10+' sigmas '+'-'*10)
    for i in range(mixture_num):
        print ('sigma[%d]=\n\t%s' % (i, sigmas[i]))

if __name__ == '__main__':
    
    mnist = []
    for i in range(10):
        mnist.append([])
    
    xs = np.genfromtxt(MNIST_traindata, delimiter=',')
    xs = xs[..., np.newaxis]

    labels = np.genfromtxt(MNIST_trainlabel, delimiter=',')

    for i in range(len(labels)):
        mnist[int(labels[i])].append(xs[i])
    # pdb.set_trace()

    for idx, xs in enumerate(mnist):
        print ('ID:', idx)
        n_data = len(xs)
        ys = [list(range(mixture_num)) for i in range(n_data)]
        is_singular = False
        is_overflow = False
        main()
        alphas_ = copy.deepcopy(alphas)
        mus_ = copy.deepcopy(mus)
        sigmas_ = copy.deepcopy(sigmas)
        all_alphas.append(alphas_)
        all_mus.append(mus_)
        all_sigmas.append(sigmas_)


