from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import operator
from sklearn.metrics import accuracy_score
# of Binary NB
# bag of words VS tf-idf
# https://blog.csdn.net/m0_37744293/article/details/78881231
# tfidf: https://www.cnblogs.com/mengnan/p/9307648.html
class customizenaivebayes1vsrest:

    # bookkeeping all classifiers
    priorprob = [[]] # 9*2
    nx = 0
    dx = 0
    c = 0
    betas=[[[]]] # dx*c*c
    betamatrix=[[[]]] # dx*c*c
    def fit(self, x, y,alpha):
        # x : numpy array n * d
        # y : multilabel indicator vectors n * k ( numpy array)
        self.nx = len(x)
        self.dx = len(x[0])
        self.c = len(y[0])
        self.betas=np.zeros((self.dx,self.c,2))
        self.betamatrix=np.zeros((self.dx,self.c,2))
        self.priorprob=np.zeros((self.c,2))
        # prior probabilities the class prob 每个类的比例

        for i in range(self.c):
            for k in range (2):
                count1=0
                count2=0
                for j in range(self.nx):
                    if y[j,i]==1 :
                        count1=count1+1
                    else:
                        count2=count2+1
                self.priorprob[i,0]=count1
                self.priorprob[i,1]=count2
        
        # betawc 每个词在每个类的个数
        for index in range(self.c): # for every class
            for k in range(2):
                if k==0:
                    for f in range(self.dx):
                        #countwords=0
                        for number in range(self.nx):
                            if y[number,index]==1:
                                self.betas[f,index,k]=self.betas[f,index,k]+x[number,f]
                else:
                    for n in range(self.c):
                        if n != index:
                            for f in range(self.dx):
                                for number in range(self.nx):
                                    if y[number,n]==1:
                                        self.betas[f,index,k]=self.betas[f,index,k]+x[number,f]
        #print(self.betas)

        # proportion of  beta
        # alpha is used here
        for i in range(self.c):
            #alpha=10
            for k in range(2):
                if k==0:
                    sumtemp=0
                    w=np.zeros(self.dx)
                    for number in range (self.nx):
                        if y[number,i]==1:
                            sumtemp += sum(x[i])# 一个类里全部单词数量和
                    for j in range(self.dx):
                        if self.betas[j,i,k]!=0:# unique words 数量
                            w[j]=w[j]+1
                    for j in range(self.dx):
                        self.betamatrix[j,i,k]=(self.betas[j,i,k]+alpha)/(sumtemp+w[j]*alpha)
                else:
                    sumtemp=0
                    w=np.zeros(self.dx)
                    for n in range(self.c):
                        if n != i:
                            for number in range (self.nx):
                                if y[number,n]==1:
                                    sumtemp += sum(x[i])# 一个类里全部单词数量和
                    for j in range(self.dx):
                        if self.betas[j,i,k]!=0:# unique words 数量
                            w[j]=w[j]+1
                    for j in range(self.dx):
                        self.betamatrix[j,i,k]=(self.betas[j,i,k]+alpha)/(sumtemp+w[j]*alpha)

    # test prediction part
    def predict(self,x):
        #print(self.betamatrix)
        ntest=len(x)
        #print(ntest)
        #dtest=len(x[0])
        result = np.ones(shape=(ntest,self.c,2))
        temp = np.ones(shape=(ntest,self.c))
        # 计算最大prob是哪一个类
        for j in range(ntest):
            for i in range(self.c):
                for k in range(2):
                    if k==0:
                        for f in range(self.dx):
                            result[j,i,k]=result[j,i,k]*(math.pow(self.betamatrix[f,i,k],x[j,f]))
                        result[j,i,k]=result[j,i,k]*self.priorprob[i,k]
                    else:
                        for f in range(self.dx):
                            result[j,i,k]=result[j,i,k]*(math.pow(self.betamatrix[f,i,k],x[j,f]))
                        result[j,i,k]=result[j,i,k]*self.priorprob[i,k]
                max_index, max_number = max(enumerate(result[j,i,:]), key=operator.itemgetter(1))
                if 0==max_index:
                    temp[j,i]=1
                else:
                    temp[j,i]=0
        #print(temp)
        return temp          
        # 或者给一个score，大于 1/9 就算这个类别，小于就不算
