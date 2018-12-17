#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import random
import copy
import pdb

SYNTHESIS_DATA = False
DEBUG_INFO = False


if SYNTHESIS_DATA:
    data = np.array((
        (0,0),
        (1,0),
        (0,1),
        (1,1),
        (2,1),
        (1,2),
        (2,2),
        (3,2),
        (6,6),
        (7,6),
        (8,6),
        (7,7),
        (8,7),
        (9,7),
        (7,8),
        (8,8),
        (9,8),
        (8,9),
        (9,9)
    ), dtype=np.float32)
    n_clusters = 2
    n_points = len(data)

else:
    samples_path = '/Users/snk/Downloads/实验/Exp1/ClusterSamples.csv'
    labels_path = '/Users/snk/Downloads/实验/Exp1/SampleLabels.csv'
    data = np.genfromtxt(samples_path, delimiter=',')
    label = np.genfromtxt(labels_path, delimiter=',', dtype=np.int8)
    n_clusters = 10
    n_points = len(label)

centers = [0] * n_clusters
clusters = [list() for i in range(n_clusters)]

def init_clusters_centers():
    picked = random.sample(range(n_points), n_clusters)
    for idx, picked_id in enumerate(picked):
        centers[idx] = data[picked_id]

    if DEBUG_INFO:
        print ('Picked Ids:')
        print (picked)
        print ('Init Centers:')
        for i,c in enumerate(centers):
            print ("%d:%s" % (i, c))

def distance_metric(x, y):
    dist = np.linalg.norm(x - y)
    return dist

def assign_cluster():
    for cluster in clusters:
        cluster.clear()

    for idx, p in enumerate(data):
        min_v = distance_metric(p, centers[0])
        min_i = 0
        for c in range(1,n_clusters):
            dist = distance_metric(p, centers[c])
            if dist < min_v:
                min_v = dist
                min_i = c
        clusters[min_i].append(idx)

def recalc_centers():
    for c in range(n_clusters): 
        if not len(clusters[c]):
            print ('Empty Cluster!')
            centers[c] = random.choice(data)
        else:
            center = 0
            for p_id in clusters[c]:
                center += data[p_id]
            center /= len(clusters[c])
            centers[c] = center

def judge_converge(last_clusters, clusters):
    for c in range(n_clusters):
        if set(last_clusters[c]) != set(clusters[c]):
            return False
    return True


def main():
    last_clusters = [list() for i in range(n_clusters)]
    iteration = 0
    init_clusters_centers()
    while True:
        assign_cluster()
        recalc_centers()
        if judge_converge(last_clusters, clusters):
            break
        last_clusters = copy.deepcopy(clusters)
        iteration += 1
        print ("Iter : %d" % iteration)

   
    if not SYNTHESIS_DATA:
        for c in range(n_clusters):
            print ('Cluster %d:' % c)
            label_cnt = [0] * 10
            for p_id in clusters[c]:
                label_cnt[label[p_id]] += 1
            for i, cnt in enumerate(label_cnt):
                print ("%d: %d" % (i, cnt))
    else:
        print (centers)
        for c in range(n_clusters):
            print ('cluster %d:\n %s' % (c, clusters[c]) )


if __name__ == '__main__':
    main()
