import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from numpy import linalg as LA

def EuclideanD(x, y):
    length = len(x)
    sum_ = 0
    for i in range(length):
        sum_ = sum_ + (x[i] - y[i])**2
    return np.sqrt(sum_)

def myKmean(data, k):
    data_cpy = data.copy()
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]
    centers = np.random.random((k, data.shape[1]))
    upcenters = centers.copy()
    convergence = False
    iterationNo = 0
    distances = np.zeros(k)

    while not convergence:
        for p in range(data_cpy.shape[0]):
            for c in range(len(centers)):
                d_p_c = EuclideanD(data_cpy[p], centers[c])
                distances[c] = d_p_c
            index_s = np.argsort(distances)
            clusters[index_s[0]].append(data_cpy[p])
            index[p] = index_s[0]
            distances = np.zeros(k)
        # UPTADE THE CENTERS
        for i in range(k):
            l_cluster = len(clusters[i])
            if l_cluster != 0:
                centers[i] = sum(clusters[i]) / l_cluster
            else:
                print("I am here")
                centers[i] = sum(clusters[i])
        # COMPARE THE CENTERS WITH THE PREVIOUS ONES
        if np.array_equal(upcenters, centers):
            convergence = True
            print("True")
        else:
            upcenters = centers.copy()
            clusters = [[] for i in range(k)]
            iterationNo += 1
            print('iterationNo = ', iterationNo)

    return index,centers,clusters