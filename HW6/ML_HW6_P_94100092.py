import pandas as pd
import numpy as np
import math
import imageio
import random
from scipy.stats import multivariate_normal
from scipy import misc
import glob
import matplotlib.pyplot as plt




np.set_printoptions(threshold=np.nan)
pd.set_option('display.expand_frame_repr', False)

data=pd.read_csv("College.csv").values
data[np.where(data[:,1]=='Yes'),1]=1
data[np.where(data[:,1]=='No' ),1]=0
ddata=data[:,2:]

def K_Means(k=2):


    iteration=100

    cl=np.copy(ddata[np.random.randint(ddata.shape[0], size=k),:])

    for tt in range(iteration):

        print("Iteration ",tt)

        ###Calculate labels

        dst=np.zeros((ddata.shape[0],k))

        for i in range(k):
            dst[:,i]=np.sum(np.power(ddata-cl[i],2),axis=1)

        lbl=np.argmin(dst,axis=1)






        ### Benchmark
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if(k==2):

            mehdi = np.c_[1-lbl, lbl].astype(int)  ##predict

            if np.average(mehdi,axis=0).reshape(2)[0]>0.5:
                mehdi=np.c_[mehdi[:,1],mehdi[:,0]]
            mansor = np.c_[1-data[:, 1], data[:, 1]].astype(int)  ##reality

            conf = np.matmul(mehdi.transpose(), mansor)

            rand_index = (conf[0, 1] + conf[1, 1]) / mehdi.shape[0]
            purity = conf[1, 1] / (conf[1,0]+conf[1,1])
            print("Purity : ",purity,"---","Rand Index",rand_index)


        var=np.zeros(k)
        for jiji in range(k):

            #p1=np.average((data[np.where(lbl==jiji), 1].reshape(-1)).astype(float))
            #print("Class ", jiji, " Average Label: ",p1)

            var[jiji]=np.asscalar(np.average(data[np.where(lbl==jiji),1].reshape(-1,1),axis=0))
            if var[jiji]>0.5:
                var[jiji]=1-var[jiji]
        print("Within Group Variance : ",np.asscalar(np.average(var)))

        ### Update class medians

        for i in range(k):
            cl[i]=np.average(   ddata[np.where(lbl==i)]     ,    axis=0  )

    return (cl,np.asscalar(np.average(var)))

def Normal(x,mio,sigma):



    #sigma=sigma.astype(float)



    soorat=math.exp(   -0.5*np.asscalar(   np.matmul(   np.matmul(    np.transpose(x-mio),np.linalg.inv(sigma)    ) , x-mio   )    )   )
    makhraj=(  np.linalg.det(sigma) * (2*math.pi)**x.shape[0]   )**0.5

    #print(-0.5*np.asscalar(   np.matmul(   np.matmul(    np.transpose(x-mio),np.linalg.inv(sigma)    ) , x-mio   )    )  ,makhraj)
    return soorat/makhraj


def GMM(Mio):

    #print(data[:,1])
    #exit()

    ddata = data[:, 2:].astype(float)

    pi0= len(np.where(data[:,1] == 0)[0])/data.shape[0]
    pi1= len(np.where(data[:,1] == 1)[0])/data.shape[0]

    #mio0=np.average(   ddata[np.where(data[:,1]==0),:].reshape(-1,ddata.shape[1]) ,  axis=0)
    #mio1=np.average(   ddata[np.where(data[:,1]==1),:].reshape(-1,ddata.shape[1]) ,  axis=0)

    mio0=Mio[0]
    mio1=Mio[1]


    #mio0=ddata[0 ,:]
    #mio1=ddata[19,:]


    sigma0 = np.random.rand(ddata.shape[1], ddata.shape[1])
    sigma1 = np.random.rand(ddata.shape[1], ddata.shape[1])
    sigma0=1/20*np.matmul(sigma0,sigma0.transpose())
    sigma1=1/20*np.matmul(sigma1,sigma1.transpose())


    for i in range(ddata.shape[1]):
        sigma0[i,i]=mio0[i]**2
        sigma1[i,i]=mio1[i]**2



    iteration=100

    for iter in range(iteration):

        #####E

        tmp = np.zeros((ddata.shape[0], 2))

        for i in range(ddata.shape[0]):
            tmp[i, 0] = pi0 * Normal(np.transpose(ddata[i]), mio0, sigma0)
            tmp[i, 1] = pi1 * Normal(np.transpose(ddata[i]), mio1, sigma1)

        tmp = tmp / tmp.sum(axis=1).reshape(-1, 1)

        predict=np.argmax(tmp,axis=1)




        print("######################")
        print("Accuracy : ",len(np.where((predict-data[:,1])==0)[0])/data.shape[0])

        mehdi = np.c_[1 - predict, predict].astype(int)  ##predict

        if np.average(mehdi, axis=0).reshape(2)[0] > 0.5:
            mehdi = np.c_[mehdi[:, 1], mehdi[:, 0]]
        mansor = np.c_[1 - data[:, 1], data[:, 1]].astype(int)  ##reality

        conf = np.matmul(mehdi.transpose(), mansor)

        rand_index = (conf[0, 1] + conf[1, 1]) / mehdi.shape[0]
        purity = conf[1, 1] / (conf[1, 0] + conf[1, 1])
        print("Purity : ", purity, "---", "Rand Index", rand_index)


        #####M

        mio0 = np.average(ddata * tmp[:, 0].reshape(-1, 1), axis=0) / pi0
        mio1 = np.average(ddata * tmp[:, 1].reshape(-1, 1), axis=0) / pi1

        lio = (ddata - np.transpose(mio0)) * np.power(tmp[:, 0], 0.5).reshape(-1, 1)
        lio = np.matmul(lio.transpose(), lio) / ddata.shape[0]
        lio = lio / pi0
        sigma0 = lio

        lio = (ddata - np.transpose(mio1)) * np.power(tmp[:, 1], 0.5).reshape(-1, 1)
        lio = np.matmul(lio.transpose(), lio) / ddata.shape[0]
        lio = lio / pi0
        sigma1 = lio

        pi0 = np.average(tmp[:, 0], axis=0)
        pi1 = np.average(tmp[:, 1], axis=0)


def PCA(dim):





    X_train=np.zeros((630,65536))
    X_test =np.zeros((630,65536))
    y_train=np.arange(30).repeat(21)
    y_test =np.arange(30).repeat(21)
    cnt=0

    for i in range(1,31):
        for j in range(1,22):
            I=i.__str__()
            J=j.__str__()
            if(i<10):I="0"+I
            if(j<10):J="0"+J
            X_train[cnt,:]=imageio.imread("Dataset/train/"+I+"_"+J+".png").reshape(1,-1)
            X_test [cnt,:]=imageio.imread("Dataset/test/" +I+"_"+J+".png").reshape(1,-1)
            cnt+=1





    mio=np.average(X_train,axis=0).reshape(1,-1)

    var=np.power(np.average(np.power(X_train-mio,2),axis=0),0.5).reshape(1,-1)
    var[np.where(var==0)]=1

    train_normalized=(X_train-mio)/var
    test_normalized=(X_test-mio)/var

    import sklearn.decomposition

    pca = sklearn.decomposition.PCA(n_components=dim)
    pca.fit(train_normalized)
    jojo=pca.transform(test_normalized)

    fofo=pca.transform(train_normalized)


    predict=np.zeros(X_test.shape[0])
    for t in range(jojo.shape[0]):
        predict[t]=np.argmin(np.sum(np.power((   fofo - jojo[t,:].reshape(1,-1)  ),2),axis=1))


    predict=((predict/21).astype(int))

    print(len(np.where(y_test-predict==0)[0])/predict.shape[0])
    return len(np.where(y_test-predict==0)[0])/predict.shape[0]


def LDA():


    X_train = np.zeros((630, 65536))
    X_test = np.zeros((630, 65536))
    y_train = np.arange(30).repeat(21)
    y_test = np.arange(30).repeat(21)
    cnt = 0

    for i in range(1, 31):
        for j in range(1, 22):
            I = i.__str__()
            J = j.__str__()
            if (i < 10): I = "0" + I
            if (j < 10): J = "0" + J
            X_train[cnt, :] = imageio.imread("Dataset/train/" + I + "_" + J + ".png").reshape(1, -1)
            X_test[cnt, :] = imageio.imread("Dataset/test/" + I + "_" + J + ".png").reshape(1, -1)
            cnt += 1

    mio = np.average(X_train, axis=0).reshape(1, -1)

    var = np.power(np.average(np.power(X_train - mio, 2), axis=0), 0.5).reshape(1, -1)
    var[np.where(var == 0)] = 1

    train_normalized = (X_train - mio) / var
    test_normalized = (X_test - mio) / var


    mio=np.zeros((30,X_train.shape[1]))
    cov=np.zeros((30,X_train.shape[1],X_train.shape[1]))

    print("aaaa")

    for i in range(30):
        ss=train_normalized[i*21:i*21+21].astype(float)
        mio[0]= np.average(ss,axis=0)
        cov[0]= np.matmul(ss.transpose(),ss)/21

    print("TA inja")

    predict=np.zeros(test_normalized.shape[0],30)

    cccc=0
    for i in range(test_normalized):
        for j in range(30):
           cccc+=1
           print(cccc)
           predict[i,j]=Normal(test_normalized[i].reshape(-1,1),mio[j].reshape(-1,1),cov[j])

    print(predict)





#K_Means(2)

def Part1():
    K_Means(2)

def Part2():
    (means,var)=K_Means(2)
    means.shape
    GMM(means.astype(float))

def Part3():
    var=np.zeros(10)
    for i in range(2,10):
        (means, varr) = K_Means(i)
        var[i]=varr

    plt.plot(range(2, 10), var[2:])
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within Variance")
    plt.title("F_test")
    plt.show()
    #print(var)


def Part4():
    xx=np.arange(1,30)
    yy=np.zeros(29)
    for i in range(1,30):
        print(i)
        yy[i-1]=PCA(i)

    plt.plot(xx, yy)
    plt.xlabel("pca's dimension")
    plt.ylabel("Accuracy ")
    plt.title("PCA")
    plt.show()

#Part1()
#Part2()
#Part3()
#Part4()