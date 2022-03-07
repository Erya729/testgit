'''
Description: regression program in ch08 of  <<machine learning in action>>  
User: zenglong@sz.tsinghua.edu.cn
Date: 2022-03-06 06:39:59
'''
import os
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font",family='KaiTi')  # 用来支持Matplotlib的汉字显示，幼圆
matplotlib.rcParams['axes.unicode_minus']=False

def loadDataSet(fileName):
    fr = open(fileName)
    numFeat = len(fr.readline().split('\t'))-1
    dataMat = []; labelMat = []
    fr = open(fileName)    # let the file pointer back to the start position
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegress(xArr, yArr):
    xMat = mat(xArr);   yMat = mat(yArr)
    xTx = xMat.T*xMat
    if fabs(linalg.det(xTx)) <0.0001: # linear algebra library in NumPy
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat.T)
    return ws

def visualizeData(xArr, yArr,ws):
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws
    titleStr = str("回归系数(%.2f,%.2f)" % (ws[0,0], ws[1,0]))
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    ax.set_title(titleStr)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0], s=5, c=[0.2,0.2,0.2])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
   # ax.plot(xCopy[:,1], yHat, c='r')
    plt.show()

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.realpath(__file__)) 
    filename = baseDir + '\\ex0.txt'
    dataMat, labelMat = loadDataSet(filename) 
    ws = standRegress(dataMat, labelMat)
    visualizeData(dataMat, labelMat, ws)