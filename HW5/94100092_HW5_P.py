import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import math
from sklearn import  tree
import xgboost as xgb
np.set_printoptions(threshold=np.nan)

train=pd.read_csv("data_train.csv").values
test =pd.read_csv("data_test.csv" ).values


def knn(xx,k):
    return np.argmax(np.bincount(( (train[:, -1])[(np.sum(np.power(train[:, 0:-1] - xx, 2), axis=1)).argsort()])[0:k]))


def part1(k) :

    cnt = 0
    for i in range(test.shape[0]):
        if (knn(test[i, 0:-1], k) == test[i, -1]): cnt = cnt + 1

    print(  "k=  "+k.__str__()+"  ,  Accuracy  :  "+(cnt / test.shape[0]).__str__()    )

def part2() :

    tree_count=15;
    tree_feature_counts=3;
    tree_max_depth=3

    clff = [None]*tree_count
    attr=((train.shape[1])*np.random.rand(tree_count,tree_feature_counts)).astype(int)

    for i in range(tree_count):
        clff[i]=tree.DecisionTreeClassifier(criterion="entropy",max_depth=tree_max_depth)
        clff[i].fit(train[:,attr[i,:]],train[:,-1])

    prdd= -1*np.ones((test.shape[0],tree_count))
    prddd= -1*np.ones(test.shape[0])

    for i in range(tree_count):
       prdd[:,i]=clff[i].predict(test[:,attr[i,:]])

    for i in range(test.shape[0]):
        prddd[i]=(np.argmax(np.bincount(prdd[i,:].astype(int))))

    print("Accuracy  :  "+((test.shape[0]-np.count_nonzero(prddd-test[:,-1]))/test.shape[0]).__str__())
    print("Confusion Matrix:")
    conf = np.zeros((prddd.shape[0],10))
    y_label= np.zeros((prddd.shape[0],10))
    conf   [np.arange(prddd.shape[0]),prddd.astype(int)]=1
    y_label[np.arange(prddd.shape[0]),test[:,-1]]=1

    print(np.matmul(conf.transpose(),y_label))


def part3():

    tree_count=10
    tree_max_depth=4
    clff = [None] * tree_count
    alpha= [None] * tree_count
    weights= 1/train.shape[0]*np.ones(train.shape[0])

    y_decod=np.zeros((train.shape[0],10))

    y_decod[np.arange(train.shape[0]),train[:,-1]]=1

    for i in range(tree_count):
        clff[i]=tree.DecisionTreeClassifier(criterion="entropy",max_depth=tree_max_depth)
        clff[i].fit(X=train[:,0:-1],y=train[:,-1],sample_weight=weights)

        predict=clff[i].predict(train[:,0:-1])
        ss=np.where( predict-train[:,-1]!=0)
        epsilon=  np.sum(weights[ss])/np.sum(weights)
        alpha[i]= math.log((1-epsilon)/epsilon)
        weights[ss]*= math.exp(alpha[i])
        #print(epsilon)

        predicton = np.zeros((test.shape[0], 10))
        for j in range(i):
            y_decod = np.zeros((test.shape[0], 10))
            y_decod[np.arange(test.shape[0]), clff[j].predict(test[:, 0:-1])] = alpha[j]
            predicton = predicton + y_decod
        acc = np.argmax(predicton, axis=1) - test[:, -1]
        print("t= "+i.__str__()+"  Accuracy  :  "+((acc.shape[0] - np.count_nonzero(acc)) / acc.shape[0]).__str__()+"  epsilon:   "+epsilon.__str__())


def part4(tree_count):


    tree_max_depth=6
    clff = [None] * tree_count
    alpha= [None] * tree_count
    weights= 1/train.shape[0]*np.ones(train.shape[0])

    y_decod=np.zeros((train.shape[0],10))

    y_decod[np.arange(train.shape[0]),train[:,-1]]=1

    for i in range(tree_count):
        clff[i]=tree.DecisionTreeClassifier(criterion="entropy",max_depth=tree_max_depth)
        clff[i].fit(X=train[:,0:-1],y=train[:,-1],sample_weight=weights)

        predict=clff[i].predict(train[:,0:-1])
        ss=np.where( predict-train[:,-1]!=0)
        epsilon=  np.sum(weights[ss])/np.sum(weights)
        alpha[i]= math.log((1-epsilon)/epsilon)
        weights[ss]*= math.exp(alpha[i])
        #print(epsilon)

    predicton = np.zeros((test.shape[0], 10))
    for j in range(tree_count):
        y_decod = np.zeros((test.shape[0], 10))
        y_decod[np.arange(test.shape[0]), clff[j].predict(test[:, 0:-1])] = alpha[j]
        predicton = predicton + y_decod
    acc = np.argmax(predicton, axis=1) - test[:, -1]
    print("Number of decision trees = "+tree_count.__str__()+"  Accuracy  :  "+((acc.shape[0] - np.count_nonzero(acc)) / acc.shape[0]).__str__())
    plt.plot(alpha)
    plt.title("M = "+tree_count.__str__()+",  Maximum tree's depth = "+tree_max_depth.__str__())
    plt.xlabel("tree's number ")
    plt.ylabel('Alpha')
    plt.show()

def part5():
    clf = xgb.XGBClassifier(objective='multi:softprop',learning_rate=0.4,max_depth=3, n_estimators=200)
    clf.fit(train[:,0:-1],train[:,-1])
    prd=clf.predict(test[:,0:-1])
    print("Accuracy : ")
    print((prd.shape[0]-np.count_nonzero(test[:,-1]-prd))/prd.shape[0])





##########################################




print("Part1 :")

part1(k=1)
part1(k=2)
part1(k=4)
part1(k=8)
part1(k=16)

print("###################")


print("Part2 : ")

part2()


print("Part3 : ")

part3()



print("Part4 : ")

part4(tree_count=5)
part4(tree_count=20)
part4(tree_count=50)
part4(tree_count=100)


print ("Part 5 : ")
part5()

