#-*- coding:utf-8 -*-
from numpy import *

"""
    功能：用于产生一些实验用的数据
"""
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#创建所给文本内的单词的集合 
def createVocabList(dataSet):
    vocabSet = set([])  #Create an empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) # "|"按位或操作 
    return list(vocabSet)

#将单词集合转化为vector形式  
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #创建全0的vector
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
        #else: print "the word: %s is not in my Vocabulary!" % word   #python 2.x

    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) #训练集中“粗鲁”类型文件的比例 
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]         #p1Num每一行的每一元素为该单词在“粗鲁”类别中出现的总次数
            p1Denom += sum(trainMatrix[i])  #粗鲁文件的所有单词总数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        p1Vect = log(p1Num/p1Denom)  #字典中的每个单词在标记为“粗鲁”的文件中出现的比例
        p0Vect = log(p0Num/p0Denom)  #字典中的每个单词在标记为“非粗鲁”的文件中出现的比例
        return p0Vect,p1Vect,pAbusive

#
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V, p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classfied as: ', classifyNB(thisDoc, p0V, p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc   = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classfied as: ', classifyNB(thisDoc, p0V, p1V,pAb))



