'''
Description: logistic regression program in ch05 of <<machine learning in action>> 
User: zenglong@sz.tsinghua.edu.cn
Date: 2022-03-03 13:01:10
'''
from cProfile import label
from cmath import exp
import time
import matplotlib.pyplot as plt
import os
from numpy import *
import matplotlib
matplotlib.rc("font",family='KaiTi')  # 用来支持Matplotlib的汉字显示，幼圆
matplotlib.rcParams['axes.unicode_minus']=False # 在figure的Axis上正常显示负号

'''
Description: read in raw data in nested list, e.g. [100,3]
User: zeng long
Date: 2022-03-04 16:25:00
param {i} filename: with full path
return feature list and label list
'''
def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)    # each sample is [feat1, feat2, label]
    for line in fr.readlines():
        lineArr = line.strip().split()  # split()默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
'''
Description: compute the sigmoid value
User: zeng long
Date: 2022-03-04 16:27:21
param {i} inX: input array
return compute values
'''
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''
Description: find the optimal coefficients with gradient ascent method
User: zeng long
Date: 2022-03-04 16:28:19
param {i} dataMatIn: samples in list or array [100,3]
param {i} classLabels: labels in a list
return optimal coefficients, i.e. w0, w1, w2
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)                 # convert into a 100*3 matrix
    labelMat = mat(classLabels).transpose()     # convert into a column vector
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))   # (n,1) ndarray
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)     # matrix dot multiply, different to array multiply which is elementwise
        error = labelMat - h
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

'''
Description: plot the fitted line
User: zeng long
Date: 2022-03-04 16:33:34
param {i} weights: (3,1) ndarray computed coefficients by gradAscent()
return void
'''
def plotBestFit(weights):
    baseDir = os.path.dirname(os.path.realpath(__file__)) 
    filename = baseDir + '\\testSet.txt'   
    dataMat, labels = loadDataSet(filename)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labels[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])            
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1) # ndarray
    y = (-weights[0]-weights[1]*x)/weights[2] 
    y = array(y).squeeze() # array() must be added, otherwise its shape is a matrix (1,60)
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

""" read in samples, visualize them w.r.t. classes, then compute and draw the fitted line"""
def plotBestFitWeights(weights, w0, w1, w2):
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('参数收敛性分析')
    # display the scatter raw points
    baseDir = os.path.dirname(os.path.realpath(__file__)) 
    filename = baseDir + '\\testSet.txt'   
    dataMat, labels = loadDataSet(filename)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labels[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])            
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    ax = fig.add_subplot(121)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1) # ndarray
    y = (-weights[0]-weights[1]*x)/weights[2] 
    y = array(y).squeeze() # array() must be added, otherwise its shape is a matrix (1,60)
    ax.plot(x, y)
    ax.set_xlabel('X1') 
    ax.set_ylabel('X2')

    # plot the coefficient curves
    x = range(0, len(w0),1)
    ax1 = fig.add_subplot(322)
    ax1.plot(x, w0)
    ax1.set_ylabel('W0')
    
    ax2 = fig.add_subplot(324)
    ax2.plot(x, w1)
    ax2.set_ylabel('W1')
    
    ax3 = fig.add_subplot(326)
    ax3.plot(x, w2)
    ax3.set_ylabel('W2')
    
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = ones(n)   # weights is a (3,) ndarray
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscentWeights1(dataMatrix, classLabels, numIter=150):
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = ones(n)   # weights is a vector
    W0=[]; W1=[]; W2=[]
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            W0.append(weights[0]); 
            W1.append(weights[1]); 
            W2.append(weights[2]); 
            alpha = 4.0/(i+j+1.0)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) 
    return weights, W0, W1, W2
'''
Description: estimate weights with stochastic ascent gradient 
User: zeng long
Date: 2022-03-06 17:32:15
param {i} dataMatrix: array, e.g. 100*3
param {i} classLabels: array, e.g. 3*1
param {i} numIter: maximum epoch
return learned weights and their learning history
'''
def stocGradAscentWeights0(dataMatrix, classLabels, numIter=10):
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = ones(n)   # weights is a vector
    W0=[]; W1=[]; W2=[]
    for j in range(numIter):
        for i in range(m):
            W0.append(weights[0]); 
            W1.append(weights[1]); 
            W2.append(weights[2]); 
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i]-h
            weights = weights + alpha * error * dataMatrix[i]
    return weights, W0, W1, W2
####################################### 以下为疝气病症预测病马的实例代码 ###################################################
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5: 
        return 1
    else: 
        return 0

def classifyVector2(inX, weights, ratio=1.0):
    prob = sigmoid(sum(inX*weights))
    negProb = 1-prob
    if negProb<0.1:
        return 1
    # adjustRatio = float(prob)/negProb*ratio
    adjustRatio = float(prob)/negProb
    if adjustRatio>1: 
        return 1
    else: 
        return 0

def colicTest():
    baseDir = os.path.dirname(os.path.realpath(__file__)) 
    frTrain = open(baseDir+'\\horseColicTraining.txt')
    frTest = open(baseDir+'\\horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    positiveSample = 0; negSample = 0
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
        if int(trainingLabels[-1])== 1:
            positiveSample += 1
        else:
            negSample += 1
    trainWeights,w0,w1,w2 = stocGradAscentWeights1(array(trainingSet),trainingLabels, 500)
    errorCount=0; numTestVec = 0.0
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        numTestVec += 1
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # if int(classifyVector(lineArr, trainWeights)) != int(currLine[21]):
        #     errorCount += 1
        if int(classifyVector2(lineArr, trainWeights, float(negSample)/positiveSample)) != int(float(currLine[21])):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec     
    print("The error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for i in range(numTests):
        errorSum += colicTest()
    print("After %d iterations, the average error rate is: %f." % (numTests, errorSum/numTests))

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.realpath(__file__)) 
    filename = baseDir + '\\testSet.txt'
    dataMat, labels = loadDataSet(filename)
    t = time.perf_counter()
    # weights = gradAscent(dataMat, labels)
    # print(f'Time cost of standard gradient:{time.perf_counter() - t:.8f}')
    # plotBestFit(weights)
    #weights = stocGradAscent0(array(dataMat), labels)
    # weights, w0, w1, w2 = stocGradAscentWeights0(array(dataMat), labels, 100)
    
    # plotBestFitWeights(weights, w0,w1,w2)
    multiTest()
    print(f'Time cost of stochastic gradient:{time.perf_counter() - t:.8f}')         
