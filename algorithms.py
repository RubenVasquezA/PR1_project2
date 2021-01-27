import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from numpy import linalg as LA
import cv2 as cv
from numpy import random
import copy


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
    "From our k-meand algo"
    "cluster has the follows: {cluster_j:[x_i]}"
    "example assigned_cluster: {0:[[1,2],[3,4]],1:[[5,6],[9,8]],...}"
    "let's compute means and covariance"
    N = data.shape[0]
    k = len(clusters)
    #clusters = {i:[] for i in range(k)}
    mu_g = np.mean(data,axis=0) #mean of overall data
    mu_c = [] #mean of each cluster
    covariance =  []
    # according to lecture mu_j and sigma_j play a role very important to compute theta
    mu_j = []
    sigma_j = []

    S_W = 0
    S_B = 0
    #mu = {i:[] for i in range(k)}, in the case we work with dict
    #covariance = {i:[] for i in range(k)}
    for c in clusters:#
        mu_c.append(np.mean(np.asarray(c),axis=0))
        covariance.append(np.cov((np.asarray(c)).T))

    #print("clusters:",len(clusters))
    #print("covariance:",len(covariance))
    #print("mu_c:",len(mu_c))

    for co in covariance:
        if len(co)!=0:
            S_W = S_W + co

    for mu in mu_c:
        if len(mu)!=0:
            S_B = S_B + np.dot((mu_g-mu_c).T,(mu_g-mu_c))
    if k==2:
        inv_S_W = np.linalg.inv(S_W)
        W = np.dot(inv_S_W,mu_c[0]-mu_c[1])
        mu_j = [np.dot(W.T,mu) for mu in mu_c]
        sigma_j = [np.dot(W.T,np.dot(co,W)) for co in covariance]
        a = sigma_j[1]*mu_j[0]-sigma_j[0]*mu_j[1]
        b = sigma_j[0]*mu_j[1]**2 - sigma_j[1]*mu_j[0]**2 + 2*sigma_j[0]*sigma_j[1]*np.log(sigma_j[1]/sigma_j[0])
        c = sigma_j[0]-sigma_j[1]
        first_term = sigma_j[1]*mu_j[0]-sigma_j[0]*mu_j[1]
        second_term = np.sqrt((a/c)**2-(b/c))
        theta= first_term + second_term
        if theta>=mu_j[0] and theta<=mu_j[1]:
            return W,theta
        else:
            theta= first_term - second_term
            return W,theta
    else:
        inv_S_W = np.linalg.inv(S_W)
        W = np.linalg.eig(inv_S_W.dot(S_B))
        # TODO
        return W

def LDAClassification(W,x,thetha):
    y = np.dot(W.T,x)
    if y>thetha:
        return 1
    else:
        return 0
    
def SVM(X,y,C=1,T=1000):

    m,n = X.shape
    Z = X*y
    I = np.eye(N)
    M = np.dot(Z.T,Z) + np.outer(y,y) + 1./C + I

    mu = 1./n * np.ones(N)
    for t in range(T):
        eta = 2./(t+2)
        grd = 2*np.dot(M,mu)
        mu += eta*(I[np.argmin(grd)]-mu)
    w = np.dot(Z,mu)
    w0 = np.dot(mu,y)
    return w,w0

def checkPatch(patch,center_x,center_y):
    "Here we would think to check if a center patch(neighborhood) has enough non-zero points"
    #TODO
    return 

def generateInputs(image,w_x=4,h_y=4):
    "Given: an masked image"
    "wanted: to extract the inputs which can be found in a certain patch"
    "inputs: an image, w_x,h_y"
    "output: x which represents the input what we want to classify later on"
    x = []
    w,h = image.shape
    for i in range(w):
        for j in range(h):
            if (j+h_y<h and j-h_y>0) and (i+w_x<w and i-w_x>0):
                if np.sum(image[i,j])==0 or np.sum(image[i,j])==1 :
                    patch=image[i-w_x:i+w_x,j-h_y:j+h_y]
                    #inputs = checkPatch(patch,i,j)
                    x.append(patch.reshape(-1,3))
    return x




 
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

#print("Task 3 (b) ...")
#images_path_original = './imgs/original/'
#images_path_mask = './imgs/mask/'

#img = cv.imread(images_path_original + 'img_t003.png')
#img_mask = cv.imread(images_path_mask + 'img_t003.png')
#it = [2]
# Color_image
#img_c = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#img_c_rs = img.reshape((-1, 3))
#img_c_rs = np.float32(img_c_rs)

#display_image("Original image", img_c)
#for k in it:
#    (clusters, label_c, centers_c) = myKmeanLoyds(img_c_rs / 255, k)
#    res_c = centers_c[label_c.flatten()]
#    result_image_c = res_c.reshape((img_c.shape))
#    display_image(str(k), result_image_c)


#W = LDA(clusters,img_c_rs / 255)
#print(W)
#X,y = SVM(img_c_rs/255,label_c)
#print(X,y)
    




