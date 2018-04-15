#-*- coding:utf-8 -*-
from numpy import *

"""
    简版SMO算法实现 
"""
#从文件加载数据 
def loadDataSet(fileName):
    dataMat = []; labelMat =[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

#获取一个ALPHA值
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

#限定Alpha值得范围[L,H] 
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#简版SMO算法 
def smoSimple(dataMatIn, classLabels,C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m ,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #fXi为预测值 
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i, :].T)) + b
            #样本xi的绝对误差Ei 
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphasIold = alphas[i].copy(); alphasJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print ("L==H"); continue
                """
                  以下流程需要看SMO算法流程才能看懂 
                     https://blog.csdn.net/luoshixian099/article/details/51227754
                """
                #eta = 2K11 - K11 - K22  
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j,:].T
                if eta >=0 :
                    print("eta >= 0");continue
                """
                   Alp2_new = Alp2_old + y2(E1 - E2) / eta
                   Alp1_new = Alp1_old + (Alp2_new - Alp2_old ) * y2 / y1
                """
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphasJold) < 0.00001):
                    print("j not moving enought")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * \
                             (alphasJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphasIold) * \
                     dataMatrix[i,:] * dataMatrix[i,:].T - \
                     labelMat[j] * (alphas[j] - alphasJold) * \
                     dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphasIold) * \
                     dataMatrix[i,:] * dataMatrix[j,:].T - \
                     labelMat[j] * (alphas[j] - alphasJold) * \
                     dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[j]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j] ) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, paris changed %d" % (iter, i, alphaPairsChanged))

        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b, alphas 

