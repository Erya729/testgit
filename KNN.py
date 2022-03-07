'''
Description: 
User: zenglong@sz.tsinghua.edu.cnim
Date: 2021-07-04 16:57:40
'''
import os
from os import listdir      # obtain file names in a directory
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
from numpy import tile
from numpy import zeros

def createDataSet():
    ''' create a dataset to test KNN algorithm '''
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
'''
Description: read a dataset from an external file
User: zeng long
Date: 2022-02-11 14:55:32
param {i} filename, where each row is a sample [1, 2, 3, label], the last number is the label
    This data is read from 'datingTestSet2.txt', each row has three features:
    1. total flight miles of a year
    2. time percent of video games
    3. weekly consumption of ice cream
return the read data matrix and label vector
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
'''
Description: visualize the data
User: zeng long
Date: 2022-02-11 16:53:19
param {i} dataMat: each row [miles, games, icecreams]
param {i} whichFig: 1 means video-ice; 2 miles-video
return void
'''
def visualizeData(dataMat, labels, whichFig):
    fig = plt.figure()      # create a figure object
    ax = fig.add_subplot(111)
    if whichFig==1:
        ax.set_xlabel('time for video games')
        ax.set_ylabel('weekly icecreams')
        ax.scatter(dataMat[:,1], dataMat[:,2], 15.0*array(labels), 15.0*array(labels))
    elif whichFig==2:
        ax.set_xlabel('total flight miles')
        ax.set_ylabel('time for video games')   
        ax.scatter(dataMat[:,0], dataMat[:,1], 15.0*array(labels), 15.0*array(labels))     
    # ax.scatter(dataMat[:,1], dataMat[:,2])
    plt.show()
'''
Description: normalize the data along each column, i.e. features
User: zeng long
Date: 2022-02-11 17:53:51
param {*} dataSetMat: each row is a sample of [1, 2, 3, label]
return normalized data matrix, ranges and mini values of each feature.
'''
def autoNorm(dataSetMat):
    minVals = dataSetMat.min(0)     # minimal value w.r.t. columns, obtain 1*3 vector
    maxVals = dataSetMat.max(0)
    ranges = maxVals - minVals                                                      
    normDataSetMat = zeros(shape(dataSetMat))
    m = dataSetMat.shape[0]
    normDataSetMat = dataSetMat - tile(minVals, (m,1))
    normDataSetMat = normDataSetMat/tile(ranges, (m,1))
    return normDataSetMat, ranges, minVals

'''
Description: classify the input feature vector inX with k-nearest method.
User: zeng long
Date: 2022-02-10 14:46:46
param {i} inX: input feature vector
param {i} dataSet: training samples, each sample [f1, f2,...,] given in a row
param {i} labels: labels of all samples
param {i} k: the k parameter of k-NN
return the label of the input feature inX.
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1))-dataSet    #tile a matrix with (dtaSetSize, 1)
    sqDiffMat = diffMat**2      # square each entry
    sqDistances = sqDiffMat.sum(axis=1) # summation along each row
    distances = sqDistances**0.5    
    sortedDistIndices = distances.argsort() #argsort is a sort method of numpy.array
    classCount = {}     # Record the k nearest samples and their labels
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
'''
Description: testing the K-NN's accurracy with given dataset.
User: zeng long
Date: 2022-02-11 18:06:34
param {i} filename: input training and testing dataset.
param {i} k: parameter of K-NN method
return void
'''
def datingClassTest(filename, k):
    hoRatio = 0.10      # ratio of data used for testing
    datingDataMat, datingLabels = file2matrix(filename)
    normDataMat, ranges, minVals = autoNorm(datingDataMat) # 1*3, row vector
    m = datingDataMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifiedRes = classify0(normDataMat[i,:], normDataMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
        if classifiedRes != datingLabels[i]:
            errorCount += 1
            print("The classified result is: %d, while the real label is: %d" % (classifiedRes, datingLabels[i]))
    print("The total error rate is: %.1f%%." % (errorCount/float(numTestVecs)*100))

'''
Description: adopt K-NN for person classification.
User: zeng long
Date: 2022-02-11 18:06:34
param {i} filename: input training and testing dataset.
return void
'''
def classifyPerson(filename):
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTime = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    inArr = array([percentTime, ffMiles, iceCream])
    datingDataMat, datingLabels = file2matrix(filename)
    normDataMat, ranges,  minVals= autoNorm(datingDataMat)
    classifiedRes = classify0((inArr-minVals)/ranges, normDataMat, datingLabels, 4)
    print ('You will probably like this person: ', resultList[classifiedRes])

def mainDating(): 
    dirname = os.path.dirname(os.path.realpath(__file__))
    filename = dirname + '\\datingTestSet2.txt'
    # dataSetMat, labels = file2matrix(filename)
    # datingClassTest(filename, 4)
    classifyPerson(filename)
############################################### end of dating person classify #################################################################################
def img2vector(filename):
    returnVec = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec

def handwritingClassTest():
    hwLabels = []
    baseDir = os.path.dirname(os.path.realpath(__file__)) 
    trainingDir = baseDir + '\\Digits\\trainingDigits'
    trainingFileList = listdir(trainingDir)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] 
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        filename = baseDir +'\\Digits\\TrainingDigits\\%s' % fileNameStr
        trainingMat[i,:] = img2vector(filename)

    testDir = baseDir + '\\Digits\\testDigits'
    testFileList = listdir(testDir)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        filename = baseDir + '\\Digits\\testDigits/%s' % fileNameStr
        vectorUnderTest = img2vector(filename)
        classifiedRes = classify0(vectorUnderTest, trainingMat, hwLabels, 4)        
        if classifiedRes != classNumStr:
            errorCount += 1 
            print("The classified result is: %d, while the real label is: %d" % (classifiedRes, classNumStr))
    print ("The total number of errors is: %d" % errorCount)
    print("The total error rate is: %.1f%%." % (errorCount/float(mTest)*100))  

def mainSimple():
    data, labels = createDataSet()
    inClass = classify0([1.0,1.4], data, labels, 3)
    print(inClass)

def mainImage():
    handwritingClassTest()

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.realpath(__file__)) 
    filename = baseDir + '\\datingTestSet2.txt'
    #filename = baseDir + '\\test1.txt'
    #dataMat, labels = file2matrix(filename)
    #visualizeData(dataMat, labels, 2)
    datingClassTest(filename, 1)
    #classifyPerson(filename)
'''
扩展任务
1. 修改datingClassTest函数，让测试数据是随机抽取情况下的错误率；
2. 请使用matplotlib绘制k-error曲线
'''