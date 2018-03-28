#-*- coding:utf-8 -*-
from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    #将一个矩阵的每一行向量相加  
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #返回distances由大到小的值对应的下标   
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), #python2  classCount.iteritems()
    key=operator.itemgetter(1),reverse=True)
    #取下标为1的元素作为排序权重，排序后翻转（因为默认的排序结果是是由小到大）  
    return sortedClassCount[0][0]

"""
    解析 datingTestSet2.txt文件 
   用1 2 3 代表not like， general like, very like

"""
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  #去掉换行符 
        listFromeLine = line.split('\t')
        returnMat[index, :] = listFromeLine[0:3]  #赋值给returnMat矩阵的第index行 
        classLabelVector.append(int(listFromeLine[-1]))  # -1下标，表示listFromeLine的最后一个元素 
        index += 1
    return returnMat, classLabelVector 

#数据归一化 new_x = (x-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0) #0 获取列的最小值  1：获取行的最小值 
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]   #获取dataSet的矩阵的行数  为1的话--获取矩阵的列数
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges,minVals

#测试
def datingClassTest():
    hoRatio = 0.10   # 留10%用作测试集 
    k = 3

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals      = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount  = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m], k)
        print "the classifier came back with:%d, the real answer is: %d" % (classifierResult,datingLabels[i])
        if(classifierResult != datingLabels[i]): errorCount += 1.0
        print "the total error rate is :%f " % (errorCount/float(numTestVecs))

#应用 
def classifyPerson():
    k = 3

    resultList  = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles     = float(raw_input("frequent flier miles earned per year?"))
    iceCream    = float(raw_input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels  = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals       = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])

    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, k)
    print "You will probably like this person: %s" % resultList[classifierResult - 1]

#将文本形式的32x32的图像转化为1x1024的vector
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

TRAININGDIGITS_PATH  =  'digits/trainingDigits'
TESTDIGIT_PATH       =  'digits/testDigits'

#手写图像识别（32x32的二值图像）
def handwritingClassTest():
    from os import listdir
    hwLabels = []  #分类标签 0-9
    trainingFileList = listdir(TRAININGDIGITS_PATH)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i:] = img2vector('%s/%s' % (TRAININGDIGITS_PATH, fileNameStr))

    testFileList = listdir(TESTDIGIT_PATH)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('%s/%s' % (TESTDIGIT_PATH, fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        
        if(classifierResult != classNumStr): 
            errorCount +=1.0
    
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))



