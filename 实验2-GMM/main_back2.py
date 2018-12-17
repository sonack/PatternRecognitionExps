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

mixture_num = 2
data_dim = 2
class_num = 2

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
    # pdb.set_trace()

        

def random_init_params():
    global alphas, mus, sigmas
    alphas = [random.random() for i in range(mixture_num)]
    alphas_sum = sum(alphas)
    alphas = [alpha/alphas_sum for alpha in alphas]
    mus = [np.random.randn(data_dim, 1) for i in range(mixture_num)]
    sigmas = []
    for i in range(mixture_num):
        # random_matrix = np.random.randn(data_dim, data_dim)
        # random_matrix = (random_matrix + random_matrix.T) / 2
        # for j in range(data_dim):
        #     random_matrix[j][j] = abs(random_matrix[i][i])
        sigma = np.random.rand(data_dim, data_dim)
        sigma = sigma.dot(sigma.T)
        sigmas.append(sigma)

    # alphas = [2/3, 1/3]
    # mus = [
    #     np.array(
    #         [
    #             [0.], [0]
    #         ]
    #     ),

    #     np.array(
    #         [
    #             [10.], [10]
    #         ]
    #     )
    # ]
    # sigmas = [
    #     np.array([
    #         [3,1.0],
    #         [1,1]
    #     ]),
    #     np.array([
    #         [2,2.0],
    #         [2,5]
    #     ])
    # ]
    # print (alphas)    
    # print (mus)    
    # print (sigmas)
    # print ('Init Params!')


# # covariance matrix
# sigma = np.array([[2.3, 0, 0, 0],
#            [0, 1.5, 0, 0],
#            [0, 0, 1.7, 0],
#            [0, 0,   0, 2]
#           ])
# # mean vector
# mu = np.array([[2],[3],[8],[10]])

# # input
# x = np.array([[2.1],[3.5],[8], [9.5]])


def pdf_multivariate_gauss(x, mu, cov):

    # pdb.set_trace()
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    if abs(np.linalg.det(cov)) < eps:
        # print ('cov is singular!')
        global is_singular
        is_singular = True
        return 0
    
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov + eps * np.eye(x.shape[0]))**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov + eps * np.eye(x.shape[0])))).dot((x-mu))
    return float(part1 * np.exp(part2))


# def norm_pdf_multivariate(x, mu, sigma):
#     size = len(x)
#     if size == len(mu) and (size, size) == sigma.shape:
#         det = np.linalg.det(sigma)
#         if det == 0:
#             raise NameError("The covariance matrix can't be singular")

#         norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.sqrt(det) )
#         x_mu = np.matrix(x - mu)
#         inv = sigma.I        
#         result = math.exp(-0.5 * (x_mu * inv * x_mu.T))
#         return norm_const * result
#     else:
#         raise NameError("The dimensions of the input don't match")



# print (pdf_multivariate_gauss(x, mu, sigma))
# print (norm_pdf_multivariate(x, mu, sigma))
def estimate_y():
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
            is_singular = False
            is_overflow = False
            # print ('ReRun')
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
    xs = np.genfromtxt(GMM1_data, delimiter=',')
    xs = xs[..., np.newaxis]
    n_data = len(xs)
    ys = [list(range(mixture_num)) for i in range(n_data)]
    is_singular = False
    is_overflow = False
    main()
    alphas_1 = copy.deepcopy(alphas)
    mus_1 = copy.deepcopy(mus)
    sigmas_1 = copy.deepcopy(sigmas)

    xs = np.genfromtxt(GMM2_data, delimiter=',')
    xs = xs[..., np.newaxis]
    n_data = len(xs)
    ys = [list(range(mixture_num)) for i in range(n_data)]
    is_singular = False
    is_overflow = False
    main()
    alphas_2 = copy.deepcopy(alphas)
    mus_2 = copy.deepcopy(mus)
    sigmas_2 = copy.deepcopy(sigmas)


    all_alphas.append(alphas_1)
    all_alphas.append(alphas_2)

    all_mus.append(mus_1)
    all_mus.append(mus_2)

    all_sigmas.append(sigmas_1)
    all_sigmas.append(sigmas_2)

    xs = np.genfromtxt(GMM1_testdata, delimiter=',')
    xs = xs[..., np.newaxis]
    classify(xs, 1)


