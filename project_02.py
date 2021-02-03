import numpy as np
import time
import matplotlib.pyplot as plt
from util.utilities import load_images_to_dict, move_color_axis
from algorithms import *
import copy
import random

def getNonBlackPixels(data: np.array):
    """ 
    description: what this function does is to throw the black pixels away 
    inputs: 
        data : the image as an array
    output:
        img_c_wb : image without black pixels
    """

    img_c_wb = []
    for px in img_c:
        if np.sum(px)!=0:
            img_c_wb.append(px)
    img_c_wb = np.asarray(img_c_wb)
    return img_c_wb

def getNeighborhoodSet(data: np.array):
    """ 
    description: what this function does is to get the neighbors of a center pixel 
    inputs: 
        data : the set of patches as an array
    output:
        X_neighborhood : the training set as an array
    """
    X_neighborhood = []
    for px in X:
        w,h,c = px.shape
        shift = (w*h -1)//2
        px_rs = px.reshape(-1,3)
        X_neighborhood.append(np.concatenate((px_rs[0:shift],px_rs[shift+1:w*h])))
    return np.asarray(X_neighborhood)

def getNewClusterFromX(data: np.array, clusters: list):
    """ 
    description: what this function does is to get a sub cluster of clusters according to data
    inputs: 
        data : the set of instances as an array (#instances,wx*hy-1,3)
        clusters : the clusters as a list
    output:
        newClusters : a newClusters  as an array
    """
    newClusters = [[] for i in range(len(clusters))]
    for ins in data:
        for px in ins:
            y = getTargetOfPixel(px,clusters)
            newClusters[y].append(px)
        
    return newClusters


def LDA(clusters: list,data: np.array):

    """ 
    description: what this function does is to perform LDA algorithm 
    inputs: 
        clusters: clusters as a list
        data : the image as an array
    output:
        W : the vector projection
        theta: the threshold 
    """
    X = data.reshape(-1,3) # (#number instances,#number features, 3) -> (#number instances * #number of feature,3)
    N = data.shape[0]
    k = len(clusters)
    #clusters = {i:[] for i in range(k)}
    mu_g = np.mean(X,axis=0) #mean of overall data
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
            S_B = S_B + np.dot((mu_g-mu).T,(mu_g-mu))
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
    
def generatePatches(image:np.array,w_x: int,h_y:int):
    """
    description: what this fuction does is given an masked image  
    we want to extract the inputs which can be found in a certain patch
    
    inputs: 
        image: an image as an array,
        wx: the width of the patch as an integer
        hy: the height of the patch as an integer
    output: 
        x : which represents the input what we want to classify later on
    """
    
    x = []
    indexs = []
    w,h,c = image.shape
    for i in range(w):
        for j in range(h):
            if i-w_x>=0 and i+w_x<w and j-h_y>=0 and j+h_y<h:
                if np.all(image[i-w_x:i+w_x+1,j-h_y:j+h_y+1]!=0) :
                    patch=image[i-w_x:i+w_x+1,j-h_y:j+h_y+1]
                    x.append(patch)
                    indexs.append([i,j])
    return x,indexs


def to_dict(clusters: list):
    """ 
    description: what this fuction does is to convert a list to a dict
    inputs: 
        clusters: clusters as a list
    output:
        dc : the clusters as a dict
    """
    lc = len(clusters)
    dc = {i:clusters[i] for i in range(lc)}
    return dc

# {0: [[0.5,0.8,0.1],[0.5,0.3,0.6]], 1: [[0.6,0.8,0.9], 2:[0.2,0.4,0.5]]}
def getTargetOfPixel(vector: np.array, clusters: list):
    """"
    description: what this function does is to get the target of a certain pixel
    inputs: 
        vector: a pixel as a array
        clusters: clusters as a list
    output: 
        target: target of the target as an array
    """
    y = None
    dc = to_dict(clusters)
    for c in dc:
        for px in dc[c]:
            if np.all(vector==px):
                y=c
                break 
    return y


def classifiersLDA(k: int, clusters: list, data: np.array):
    
    """ 
    description: what this function does is to obtain different classifiers 
    inputs: 
        clusters: clusters as a list
        data : the image as an array
    output:
        W : the vector projection
        theta: the threshold 
        
    comment: W,theta come from LDA algorithm
    """
        
    clusterCopy = copy.deepcopy(clusters)
    cluster1 = clusterCopy[k]
    clusterCopy.remove(cluster1)
    cluster2 = clusterCopy
    c2 = []
    for c in cluster2:
        c2 = c2 + c
    finalCluster=[]
    finalCluster.append(cluster1)
    finalCluster.append(c2)
    W,theta = LDA(finalCluster,data)
    return W,theta


def classifierSVMPolyKernel(k: int, clusters: list, data: np.array, centers: np.array):
    
    """ 
    description: what this function does is to obtain different classifiers 
    inputs: 
        clusters: clusters as a list
        data : the image as an array
    output:
        mu 
        
    comment: mu come from SVM algorithm
    """
        
    clusterCopy = copy.deepcopy(clusters)
    cluster1 = clusterCopy[k]
    clusterCopy.remove(cluster1)
    cluster2 = clusterCopy
    c2 = []
    for c in cluster2:
        c2 = c2 + c
    finalCluster=[]
    finalCluster.append(cluster1)
    finalCluster.append(c2)
    # [1,0,2,1,0] -> k=0 (vs the rest) -> y = [-1,1,-1,-1,-1]
    # k = 1 (vs rest) -> y = [1,-1,-1,1,-1]
    # k = 2 (vs rest) ->  y=[-1,-1,1,-1,-1]
    y = fromMulti2binary(k,np.array([getTargetOfPixel(px,finalCluster) for px in centers ]))
    m = trainL2SVMPolyKernel(data,y,d=3,b=2.,C=1.,T=1000)
    s = np.where(m>0)[0]
    XS = X[:,s]
    ys = y[s]
    ms = m[s]
    w0 = np.dot(ys,ms)

    return XS,ys,ms,w0

def trainL2SVMPolyKernel(X, y, d, b=1., C=1., T=1000):
    m, N = X.shape
    I = np.eye(N)
    Y = np.outer(y,y)
    K = (b + np.dot(X.T, X))**d
    M = Y * K + Y + 1./C*I
    mu = np.ones(N) / N
    for t in range(T):
        eta = 2./(t+2)
        grd = 2 * np.dot(M, mu)
        mu += eta * (I[np.argmin(grd)] - mu)
    return mu


def fromMulti2binary(k:int,vector: np.array):
    """
    description: what this function does is to convert a multiclass vector to binary
    example: if vector=[1,2,3,1,0] and k=0 then the response will be vector=[-1,-1,-1,-1,1]
    inputs: 
        k : the current cluster as a integer
        vector: the target as an array
    output:
        cvector: converted vector as an array
    """
    l = vector.shape[0]
    y = np.ones(l)
    for i in range(l):
        if vector[i]!=k:
            y[i] = -1
    return y


def LDAClassification(W: np.array,x:np.array,thetha:float,k: int):
    y = np.dot(W.T,x)
    if y>=thetha:
        return k #correct classification 
    else:
        return -1

def L2SVMPolyKernelClassification(x, XS, ys, ms, w0, d, b=1.):
    if x.ndim == 1:
        x = x.reshape(len(x),1)
        k = (b + np.dot(x.T, XS))**d
    return np.sum(k * ys * ms, axis=1) + w0


def fitClassifiers(method: str, clusters: list, data: np.array,centers: np.array):
    """ 
    description: what this function does is to obtain different classifiers 
    inputs: 
        method: method could be either LDA or SVM as a string
        clusters: clusters as a list
        data : the image as an array
    output:
        classifiers: a list of classifiers as list
    """
    if method=='LDA':
        classifiers = {i:[] for i in range(len(clusters))}
        l = len(clusters)
        for k in range(l):
            if len(clusters[k])!=0:
                W, theta= classifiersLDA(k=k,clusters=clusters,data=data)
                classifiers[k].append(W)
                classifiers[k].append(theta)
    if method=='SVM':
        classifiers = {i:[] for i in range(len(clusters))}
        l = len(clusters)
        for k in range(l):
            if len(clusters[k])!=0:
                XS,ys,ms,w0 = classifierSVMPolyKernel(k=k,clusters=clusters,data=data,centers=centers)
                classifiers[k].append(XS)
                classifiers[k].append(ys)
                classifiers[k].append(ms)
                classifiers[k].append(w0)
    return classifiers
        


# ## we can build up the classifiers according to clusters we have assigned


if __name__ == "__main__":

    # keys are file names and values are the images
    masks: dict = load_images_to_dict('./imgs/mask/')
    imgs: dict = load_images_to_dict('./imgs/original')

    # masks and images have the same filename
    img_names = list(imgs.keys())
    mask_names = list(masks.keys())
    print(img_names)
    print(mask_names)
    print(f"image shape: {imgs[img_names[0]].shape}")

    # apply masks to images
    masked = {}
    for k in imgs.keys():
        # move color dim to front: H*W*color -> color*H*W
        img = move_color_axis(imgs[k], -1, 0)
        msk = masks[k]
        # apply mask
        mskd = img*msk
        # reshape to H*W*color
        mskd = move_color_axis(mskd, 0, -1)
        masked[k] = mskd


    
    masked_image = masked['img_t003.png'] #we are using the loyds algo in here
    img_c = masked_image.reshape((-1, 3))
    img_c = getNonBlackPixels(img_c) #it is considering the bounderies 
    it = [3]
    # centers = [[1,4,5],[4,7,3],[6,5,5]] , clusters =[[[1,2,3],[4,5,3],...]], [[1,4,5,],...], [[5,4,6]]]
    for k in it:
        (clusters, label_c, centers_c) = myKmeanLoyds(img_c, k) 
        res_c = centers_c[label_c.flatten()]
        result_image_c = res_c.reshape((img_c.shape))

    X,indexsCenters = generatePatches(masked_image,2,2) #patches where all are distinct than zero
    X,indexsCenters = np.asarray(X),np.asarray(indexsCenters)
    #print(img_c.shape,X.shape)
    X = getNeighborhoodSet(X) #this what we thought if u want u can print it out 
    # [[1,2,3],[4,5,6],[7,8,9]]  -> [[1,2,3,4,6,7,8,9]] -> X = (#sample,#number features, 3)
    # shuffling proccess
    nsamples = 100
    split = int(0.8*nsamples)
    listIndexs = [i for i in range(X.shape[0])]

    #Xselected,indexSelected = X[listIndexs],indexsCenters[listIndexs] 
    # indexCenter =[[1,2],[3,4],...]
    random.shuffle(listIndexs)
    XShuffle,indexsCentersShuffle = X[listIndexs],indexsCenters[listIndexs]
    XTraining,indexsCentersTraining = XShuffle[:split],indexsCentersShuffle[:split]
    XTesting,indexsCentersTesting = XShuffle[split:],indexsCentersShuffle[split:]
    # centertrinig = [[0.3,0.5,0.6],[0.6,0.9,0.5]] 
    CentersTraining = np.array([masked_image[index[0],index[1]] for index in indexsCentersTraining])
    targetTraining = np.array([getTargetOfPixel(px,clusters) for px in CentersTraining ])
    # e1 -> [[],[],[],[],[],..,[]]
    # e2 -> [[],[],[],[],[],..,[]]
    # ...
    # e80 -> [[],[],[],[],...,[]] 
    # y = array([1,2,0,...,0]) (80,)
    # here we want to get a new cluster according to our training data: 
    newClusters = getNewClusterFromX(XTraining,clusters)
    print(newClusters)
    # To fit the LDA classifiers       
    LDAClassifiers = fitClassifiers('LDA',newClusters, XTraining,CentersTraining)
    print("Here the classifiers",LDAClassifiers)
# ## Here we generate the inputs from the masked image i.e the we take some patches which the pixel is white + its own neighborhood out 

# ## Here we gather the classifiers, for both method LDA and SVM
    dictClassification = {i:{c:[] for c in range(len(LDAClassifiers))} for i in range(len(CentersTraining))}
    dictAcc = np.array([0 for c in range(len(LDAClassifiers))])
# Classification and perfomance part:
    for i in range(len(CentersTraining)):
        for c in LDAClassifiers:
            W = LDAClassifiers[c][0]
            theta=LDAClassifiers[c][1]
            classificationLDA = LDAClassification(W,CentersTraining[i],theta,c) 
            dictClassification[i][c].append(classificationLDA)
            if classificationLDA==targetTraining[i]:
                dictAcc[c] = dictAcc[c] + 1 
    
    dictAcc = dictAcc/len(CentersTraining)
    print(dictAcc)
    # TODO 
    #SVMWithPolyClassifiers = fitClassifiers('SVM',newClusters,Xtraining,CentersTraining)
    

