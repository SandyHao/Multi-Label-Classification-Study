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
class customizenaivebayes:

    # bookkeeping all classifiers
    priorprob = []
    nx = 0
    dx = 0
    c = 0
    arrange = []
    betas=[[]]
    betamatrix=[[]]
    def fit(self, x, y):
        # x : numpy array n * d
        # y : multilabel indicator vectors n * k ( numpy array)
        self.nx = len(x)
        print(self.nx)
        self.dx = len(x[0])
        self.c = len(y[0])
        self.betas=np.zeros((self.dx,self.c))
        self.betamatrix=np.zeros((self.dx,self.c))
        # prior probabilities the class prob 每个类的比例
        for i in range(self.c):
            count=0
            for j in range(self.nx):
                if y[j,i]==1 :
                    count=count+1
            self.priorprob.append(count)
        
        # betawc 每个词在每个类的个数
        for index in range(self.c): # for every class
            for f in range(self.dx):
                #countwords=0
                for number in range(self.nx):
                    if y[number,index]==1:
                        self.betas[f,index]=self.betas[f,index]+x[number,f]
                #self.betas[f,index]=countwords
        print(self.betas)

        # proportion of  beta
        # alpha is used here
        for i in range(self.c):
            sumtemp=0
            alpha=10000
            w=np.zeros(self.dx)
            for number in range (self.nx):
                if y[number,i]==1:
                    sumtemp += sum(x[i])# 一个类里全部单词数量和
            for j in range(self.dx):
                if self.betas[j,i]!=0:# unique words 数量
                    w[j]=w[j]+1
            for j in range(self.dx):
                self.betamatrix[j,i]=(self.betas[j,i]+alpha)/(sumtemp+w[j]*alpha)

    # test prediction part
    def predict(self,x):
        print(self.betamatrix)
        ntest=len(x)
        print(ntest)
        #dtest=len(x[0])
        result = np.ones(shape=(ntest,self.c))
        
        # 计算最大prob是哪一个类
        for j in range(ntest):
            for i in range(self.c):
                for f in range(self.dx):
                    result[j,i]=result[j,i]*(math.pow(self.betamatrix[f,i],x[j,f]))
                result[j,i]=result[j,i]*self.priorprob[i]
            max_index, max_number = max(enumerate(result[j,:]), key=operator.itemgetter(1))
            for k in range(self.c):
                if k==max_index:
                    result[j,k]=1
                else:
                    result[j,k]=0
        print(result)
        return result            
        # 或者给一个score，大于 1/9 就算这个类别，小于就不算
