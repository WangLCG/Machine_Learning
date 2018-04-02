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
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        p1Vect = p1Num/p1Denom  #change to log()
        p0Vect = p0Num/p0Denom  #change to log()
        return p0Vect,p1Vect,pAbusive



