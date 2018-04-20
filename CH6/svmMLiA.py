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
            #fXi为预测值: <Alph  y> * <xi, x> + b = w.T * x + b 
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

class opStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        #eCache保存计算出来的值，第一列为值是否有效标志，第二列为值
        self.eCache = mat(zeros((self.m, 2)))

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]
    #.A运算转为array类型
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            #取最大的abs(E1-E2)来近似最大化步长 
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    #oS.alphas[i] < C  或者 oS.alphaPairsChanged[i] > 0 才能进入if条件分支，一开始oS.alphas[i]全为0，满足oS.alphas[i] < C条件 
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)
        alphasIold = oS.alphas[i].copy()
        alphasJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if L==H : print("L == H");return 0

        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i, :].T - oS.X[j,:] * oS.X[j,:].T
        if(eta >= 0): print("eta >= 0");return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei- Ej)/eta
        #裁剪过的alphas[j]符合边界约束条件  
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)

        if(abs(oS.alphas[j] - alphasJold) < 0.00001):
            print("j not moving enough");return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphasJold - oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphasIold) * oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphasJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphasIold) * oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphasJold) * oS.X[j,:] * oS.X[j,:].T

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):oS.b = b2
        else:oS.b = (b1 + b2)/2.0
        return 1
    else:return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
    oS   = opStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    #在整个样本集合非边界集上(0 < alpha_i < C)来回切换，寻找违反KKT条件的alpha_i作为第一个变量 
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d, i:%d, paris changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False
        elif(alphaPairsChanged == 0):entireSet = True
        print("iteration number:%d " % iter)
    return oS.b, oS.alphas

#计算W
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i,:].T)
        #实数与矩阵相乘--实数与矩阵的每个元素相乘 
    return w