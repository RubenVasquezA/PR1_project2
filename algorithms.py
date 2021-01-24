import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from numpy import linalg as LA
import cv2 as cv
from numpy import random
import copy


images_path = './imgs/original/'

def display_image(name, img):
    img = np.array(img).clip(min=0, max=255).astype(np.uint8)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def EuclideanD(x, y):
    length = len(x)
    sum_ = 0
    for i in range(length):
        sum_ = sum_ + (x[i] - y[i])**2
    return np.sqrt(sum_)

def myKmeanLoyds(data, k):
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
    centers = np.uint8(centers * 255)
    index = np.uint8(index)
    return clusters,index, centers

def assignedCluster(data,k):
    clusters = {i:[] for i in range(k)}
    for j in range(data.shape[0]):
        random_index = random.randint(k) 
        clusters[random_index].append(j)
    return clusters

def assignedCenters(clusters,data):
    k = len(clusters)
    centers = []
    for c in clusters:
        if len(clusters[c])!=0:
           centers.append(data[clusters[c]].mean(axis=0))
        else:
           centers.append([0 for i in range(data.shape[1])])
    return np.asarray(centers)

def findCluster(clusters,instance):
    for c in clusters:
        if instance in clusters[c]:
            print(c)
            return c

def cw(clusters,centers,data,index):
    clustersCopy = copy.deepcopy(clusters)
    for c in clustersCopy:
        clustersCopy[c].append(index)
    #print("inside of the function:",clustersCopy)
    # to find the minimun 
    E = []
    for i in range(centers.shape[0]):
        d = []
        for c in clustersCopy:
            d.append(LA.norm(data[clustersCopy[c]]-centers[i]))
        E.append(LA.norm(np.asarray(d)))
    return np.argsort(E)[0]



def labeling(clusters,data):
    k = len(clusters)
    labels = np.zeros(data.shape[0])
    for j in range(data.shape[0]):
        for c in clusters:
            if j in clusters[c]:
                labels[j] = c
    return labels


def myKmeanHartigan(data, k):
    assigned_cluster = assignedCluster(data,k)
    #print(assigned_cluster)
    centers = assignedCenters(assigned_cluster,data)
    #print(centers)
    convergence = False
    iterationNo=0
    while not convergence:
        convergence = True
        for j in range(data.shape[0]):
            #print("here assigned_cluster:",assigned_cluster)
            i_index = findCluster(assigned_cluster,j)
            print("here i_index,j:",i_index,j)
            assigned_cluster[i_index].remove(j)
            centers = assignedCenters(assigned_cluster,data)
            w_index = cw(assigned_cluster,centers,data,j)
            if w_index != i_index:
                convergence = False
            assigned_cluster[w_index].append(j)
            centers = assignedCenters(assigned_cluster,data)
            iterationNo += 1
            print('iterationNo = ', iterationNo)
    label = labeling(assigned_cluster,data)
    centers = np.uint8(centers * 255)
    label = np.uint8(label)
    return assigned_cluster,label,centers

def LDA(clusters,data):
    # From our k-meand algo
    # cluster has the follows: {"cluster_j":[x_i]}
    # example assigned_cluster: {"0":[[1,2],[3,4]],"1":[[5,6],[9,8]],...}
    # let's compute means and covariance:

    N = data.shape[0]
    k = len(clusters)
    #clusters = {i:[] for i in range(k)}
    mu_g = np.mean(data,axis=0) #mean of overall data
    mu_c = [] #mean of each cluster
    covariance =  []
    S_W = 0
    S_B = 0
    #mu = {i:[] for i in range(k)}, in the case we work with dict
    #covariance = {i:[] for i in range(k)}
    for c in clusters:
        print("here")
        mu_c.append(np.mean(np.asarray(c),axis=0))
        covariance.append(np.cov((np.asarray(c)).T))

    print("clusters:",len(clusters))
    print("covariance:",len(covariance))
    print("mu_c:",len(mu_c))

    for co in covariance:
        if len(co)!=0:
            S_W = S_W + co

    for mu in mu_c:
        if len(mu)!=0:
            S_B = S_B + np.dot((mu_g-mu_c).T,(mu_g-mu_c))

    inv_S_W = np.linalg.inv(S_W)
    eig_vals, eig_vecs = np.linalg.eig(inv_S_W.dot(S_B))

    return eig_vals,eig_vecs

            

#here some thoughts: (this piece of code makes sense, in fact it converge quite straight forward)
# data = np.array([[1,2,3,5,6],[3,5,7,7,6],[2,6,8,4,3],[1,2,5,1,6],[8,9,10,9,8],[1,4,3,6,5],[2,3,4,5,4]])
# clusters,label,centers = myKmeanHartigan(data, 2)
# print(clusters,label,centers)

# but here it turns out it does not converge :( 
# print("Task 3 (b) ...")
# img = cv.imread(images_path + 'img_t003.png')
# it = [2]
# img_c = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img_c_rs = img.reshape((-1, 3))
# img_c_rs = np.float32(img_c_rs)
# print("here the shape:",img_c_rs.shape)
# display_image("Original image", img_c)
# for k in it:
#     (assigned_cluster_c,label_c, centers_c) = myKmeanHartigan(img_c_rs / 255, k)
#     res_c = centers_c[label_c.flatten()]
#     result_image_c = res_c.reshape((img_c.shape))
#     display_image(str(k), result_image_c)

# if you run this part it will work (cause it converges):) 

print("Task 3 (b) ...")
img = cv.imread(images_path + 'img_t003.png')
it = [2]
    # Color_image
img_c = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_c_rs = img.reshape((-1, 3))
img_c_rs = np.float32(img_c_rs)
display_image("Original image", img_c)
for k in it:
    (clusters, label_c, centers_c) = myKmeanLoyds(img_c_rs / 255, k)
    res_c = centers_c[label_c.flatten()]
    result_image_c = res_c.reshape((img_c.shape))
    display_image(str(k), result_image_c)


eig_vals,eig_vecs = LDA(clusters,img_c_rs / 255)
print(eig_vals,eig_vecs)

    
        



