import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


np.random.seed(1)


##Data Loading And conversion

whole = np.loadtxt("wifi_data.txt")
rodo = np.c_[whole,np.zeros(whole.shape[0]),np.zeros(whole.shape[0]),np.zeros(whole.shape[0]),np.zeros(whole.shape[0])]
(rodo[:,8:9])   [(rodo[:,7:8]==1)]=1
(rodo[:,9:10])  [(rodo[:,7:8]==2)]=1
(rodo[:,10:11]) [(rodo[:,7:8]==3)]=1
(rodo[:,11:12]) [(rodo[:,7:8]==4)]=1
whole = rodo


##Data for Class 1 and 3

ind=np.where(np.any(np.c_[whole[:,8:9],whole[:,10:11]],axis=1))[0]
np.random.shuffle(ind)
train_data = ind[0:int (ind.shape[0]*(4/5))]
test_data  = ind[int (ind.shape[0]*(4/5)):]
X_train_13 = whole[train_data, 0:7]
Y_train_13 = np.c_[ whole[train_data, 8:9] , whole[train_data,10:11] ]
X_test_13 = whole[test_data, 0:7]
Y_test_13 = np.c_[whole[test_data, 8:9], whole[test_data, 10:11]]





##Data for all 4 Classes

ind = whole
np.random.shuffle(ind)
X_train_1234 = whole[0:int (ind.shape[0]*(4/5)), 0:7]
Y_train_1234 = whole[0:int (ind.shape[0]*(4/5)), 8:12]
X_test_1234  = whole[int (ind.shape[0]*(4/5)):, 0:7]
Y_test_1234  = whole[int (ind.shape[0]*(4/5)):, 8:12]


# Logistic Regression Functions

def Batch_Gradient_Descend_Logistic_Regression(X,Y,W,etha,landa):

    jojo = np.exp(np.matmul(X, W))
    jojo = jojo / jojo.sum(axis=1, keepdims=True)

    W+=(np.matmul(X.transpose(),  Y-jojo)-landa*2*W)*etha

    return W

def Logistic_Regression_Loss(X,Y,W,landa):


    jojo = np.exp(np.matmul(X, W))
    jojo = jojo / jojo.sum(axis=1, keepdims=True)
    jojo = np.log(jojo)

    return (-1*np.asscalar(np.multiply(jojo,Y).sum())/jojo.shape[0])+landa*np.asscalar(np.matmul(W.transpose(),W).trace())

def dummy_Loss(X, Y, W):
    fofo = np.matmul(X, W).argmax(axis=1) - Y.argmax(axis=1)
    return len(np.where(fofo == 0)[0])

def Logistic_confusion(X,Y,W):

    jojo = np.exp(np.matmul(X,W))
    jojo=jojo/np.sum(jojo,axis=1).reshape(-1,1)
    predict=(jojo==jojo.max(axis=1)[:,None]).astype(int)
    return np.matmul(Y.transpose(),predict)




    jojo/np.sum(jojo,axis=1).reshape(200,1)
    return jojo

def logistic_Regression(X_train, Y_train, X_test, Y_test, landa=0):

    np.random.seed(1)

    chunk_size=10000
    step=100
    etah=   0.000000001
    #etah = 0.000001
    W = np.random.random((7, Y_train.shape[1])) * 0.0001


    loss = np.zeros(step+1)
    dummy_loss = np.zeros(step+1)
    loss[0] = Logistic_Regression_Loss(X_train, Y_train, W, landa)
    dummy_loss[0] = dummy_Loss(X_train, Y_train, W)

    for i in range((step+2)*chunk_size):

        W = Batch_Gradient_Descend_Logistic_Regression(X_train, Y_train, W, etah, landa)

        if (i % chunk_size == 0):

            # etha*=0.97
            loss[int(i / chunk_size)] = Logistic_Regression_Loss(X_train, Y_train, W, landa)
            dummy_loss[int(i / chunk_size)] = dummy_Loss(X_train, Y_train, W)
            print(int(i).__str__() + " :  Loss = " + loss[int(i / chunk_size)].__str__() + "  Correct prediction's count = " + dummy_loss[int(i / chunk_size)].__str__())
            if (i == step*chunk_size): break

    plt.plot(np.arange(101), loss, label="error_train")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


    print("Accuracy on Train Data : " + (dummy_Loss(X_train, Y_train, W) / Y_train.shape[0]).__str__())

    print("Confusion Matrix on Train : ")
    print(Logistic_confusion(X_train, Y_train, W).__str__())

    print("Accuracy on Test Data : " + (dummy_Loss(X_test, Y_test, W) / Y_test.shape[0]).__str__())

    print("Confusion Matrix on Test : ")
    print(Logistic_confusion(X_test, Y_test, W).__str__())

    print(" W  :  ")
    print(W)


# Bayesian Classifier Functions

def Baysian_Maximum_Likelihood_Estimation(X):
    mio = np.average(X,axis=0)
    S=X-mio
    S=np.matmul(S.transpose(),S)
    return (mio,S/X.shape[0])

def Naive_Baysian_Maximum_Likelihood_Estimation(X):
    mio = np.average(X, axis=0)
    S=X-mio
    S=np.multiply(S,S)
    return (mio,np.sum(S,axis=0)/X.shape[0])

def X_if_C_MND(mio,Sigma,X):
    Sigma_inv=np.linalg.inv(Sigma)
    #return ((((2*np.pi)**X.shape[1])*np.linalg.det(Sigma))**(-1/2))*np.exp(-1/2*np.matmul(np.matmul((X-mio).transpose(),Sigma_inv),X-mio))
    #solo=np.exp(-1/2*np.sum(np.multiply(,np.identity(X.shape[0]))),axis=1)
    solo=np.matmul(np.matmul(X - mio, Sigma_inv), (X - mio).transpose())
    solo=np.multiply(solo,np.identity(solo.shape[0])).sum(axis=1)
    return ((((2*np.pi)**X.shape[1])*np.linalg.det(Sigma))**(-1/2))*np.exp(-1/2*solo)

def X_if_C_MND_Naive(mio,variance,X):

   return np.product( np.multiply(   (   (2*np.pi*np.multiply(variance,variance))**(-1/2)  ) ,   np.exp((-1/2)*np.multiply(X-mio,X-mio)) ) ,axis=1)


# Homework parts

def Part_1():

    print("\n\n##############Part1   :   \n")
    logistic_Regression(X_train_13,Y_train_13,X_test_13,Y_test_13)

def Part_2():

    print("\n\n##############Part2   :   \n")

    print("\n\n###  landa=2 :\n")
    logistic_Regression(X_train_13,Y_train_13,X_test_13,Y_test_13,landa=2)

    print("\n\n###  landa=1 :\n")
    logistic_Regression(X_train_13,Y_train_13,X_test_13,Y_test_13,landa=1)

    print("\n\n###  landa=0.5 :\n")
    logistic_Regression(X_train_13, Y_train_13, X_test_13, Y_test_13,landa=0.5)

def Part_3():

    print("\n\n##############Part2   :   \n")
    logistic_Regression(X_train_1234,Y_train_1234,X_test_1234,Y_test_1234)

def Part_4():

    print("\n\n######## Part4   :\n")

    X_train=X_train_13
    Y_train=Y_train_13
    X_test=X_test_13
    Y_test=Y_test_13

    X_train_class1 = X_train[np.where(Y_train[:, 0] == 1), :].reshape(-1, 7)
    X_train_class3 = X_train[np.where(Y_train[:, 1] == 1), :].reshape(-1, 7)


    ################Bayes


    print("\n\n############   Bayesian Classifier   :  \n")

    MLE_class1=Baysian_Maximum_Likelihood_Estimation(X_train_class1)
    MLE_class3=Baysian_Maximum_Likelihood_Estimation(X_train_class3)

    print("\n\n###MLE_class1   :")
    print("\n## mio :")
    print(MLE_class1[0])
    print("\n### covariances  :  ")
    print(MLE_class1[1])

    print("\n\n###MLE_class3   :")
    print("\n## mio :")
    print(MLE_class3[0])
    print("\n### covariances  :  ")
    print(MLE_class3[1])

    prediction_pr=np.c_[X_if_C_MND(MLE_class1[0],MLE_class1[1],X_test),X_if_C_MND(MLE_class3[0],MLE_class3[1],X_test)]
    prediction = (prediction_pr == prediction_pr.max(axis=1)[:, None]).astype(int)

    confusion = np.matmul(prediction.transpose(), Y_test)
    print("\nConfusion Matrix on Test Data    :")
    print(confusion)
    print("\nAccuracy on Test :" + (confusion.trace() / Y_test.shape[0]).__str__())


    ##############Naive Bayes


    print("\n\n############  Naive  Bayesian Classifier   :  \n")

    NMLE_class1 = Naive_Baysian_Maximum_Likelihood_Estimation(X_train_class1)
    NMLE_class3 = Naive_Baysian_Maximum_Likelihood_Estimation(X_train_class3)


    print("\n\n###NMLE_class1   :")
    print("\n## mio :")
    print(NMLE_class1[0])
    print("\n### variances  :  ")
    print(NMLE_class1[1])

    print("\n\n###NMLE_class3   :")
    print("\n## mio :")
    print(NMLE_class3[0])
    print("\n### variances  :  ")
    print(NMLE_class3[1])


    prediction_pr = np.c_[X_if_C_MND_Naive(NMLE_class1[0], NMLE_class1[1], X_test), X_if_C_MND_Naive(NMLE_class3[0], NMLE_class3[1], X_test)]
    prediction = (prediction_pr == prediction_pr.max(axis=1)[:, None]).astype(int)

    confusion = np.matmul(prediction.transpose(), Y_test)
    print("\nConfusion Matrix on Test Data    :")
    print(confusion)
    print("\nAccuracy on Test :" + (confusion.trace() / Y_test.shape[0]).__str__())

def Part_5():

    print("\n\n###########  Part5   :   ")

    X_train = X_train_1234
    Y_train = Y_train_1234
    X_test = X_test_1234
    Y_test = Y_test_1234

    X_train_class1 = X_train[np.where(Y_train[:, 0] == 1), :].reshape(-1, 7)
    X_train_class2 = X_train[np.where(Y_train[:, 1] == 1), :].reshape(-1, 7)
    X_train_class3 = X_train[np.where(Y_train[:, 2] == 1), :].reshape(-1, 7)
    X_train_class4 = X_train[np.where(Y_train[:, 3] == 1), :].reshape(-1, 7)



    ################Bays

    print("\n######   Bayesian Classifier\n")



    MLE_class1=Baysian_Maximum_Likelihood_Estimation(X_train_class1)
    MLE_class2=Baysian_Maximum_Likelihood_Estimation(X_train_class2)
    MLE_class3=Baysian_Maximum_Likelihood_Estimation(X_train_class3)
    MLE_class4=Baysian_Maximum_Likelihood_Estimation(X_train_class4)

    print("\n\n###MLE_class1   :")
    print("\n## mio :")
    print(MLE_class1[0])
    print("\n### covariances  :  ")
    print(MLE_class1[1])

    print("\n\n###MLE_class2   :")
    print("\n## mio :")
    print(MLE_class2[0])
    print("\n### covariances  :  ")
    print(MLE_class2[1])

    print("\n\n###MLE_class3   :")
    print("\n## mio :")
    print(MLE_class3[0])
    print("\n### covariances  :  ")
    print(MLE_class3[1])

    print("\n\n###MLE_class4   :")
    print("\n## mio :")
    print(MLE_class4[0])
    print("\n### covariances  :  ")
    print(MLE_class4[1])


    prediction_pr=np.c_[X_if_C_MND(MLE_class1[0],MLE_class1[1],X_test),X_if_C_MND(MLE_class2[0],MLE_class2[1],X_test),X_if_C_MND(MLE_class3[0],MLE_class3[1],X_test),X_if_C_MND(MLE_class4[0],MLE_class4[1],X_test)]
    prediction = (prediction_pr == prediction_pr.max(axis=1)[:, None]).astype(int)

    confusion=np.matmul(prediction.transpose(), Y_test)
    print("\nConfusion Matrix on Test Data    :")
    print(confusion)
    print("\nAccuracy on Test :"+(confusion.trace()/Y_test.shape[0]).__str__())


    ##############Naive Bays

    print("\n\n######  Naive  Bayesian Classifier\n")

    NMLE_class1 = Naive_Baysian_Maximum_Likelihood_Estimation(X_train_class1)
    NMLE_class2 = Naive_Baysian_Maximum_Likelihood_Estimation(X_train_class2)
    NMLE_class3 = Naive_Baysian_Maximum_Likelihood_Estimation(X_train_class3)
    NMLE_class4 = Naive_Baysian_Maximum_Likelihood_Estimation(X_train_class4)

    print("\n\n###NMLE_class1   :")
    print("\n## mio :")
    print(NMLE_class1[0])
    print("\n### variances  :  ")
    print(NMLE_class1[1])

    print("\n\n###NMLE_class2   :")
    print("\n## mio :")
    print(NMLE_class2[0])
    print("\n### variances  :  ")
    print(NMLE_class2[1])

    print("\n\n###NMLE_class3   :")
    print("\n## mio :")
    print(NMLE_class3[0])
    print("\n### variances  :  ")
    print(NMLE_class3[1])

    print("\n\n###NMLE_class4   :")
    print("\n## mio :")
    print(NMLE_class4[0])
    print("\n### variances  :  ")
    print(NMLE_class4[1])


    prediction_pr = np.c_[X_if_C_MND_Naive(NMLE_class1[0], NMLE_class1[1], X_test), X_if_C_MND_Naive(NMLE_class2[0], NMLE_class2[1], X_test),X_if_C_MND_Naive(NMLE_class3[0], NMLE_class3[1], X_test),X_if_C_MND_Naive(NMLE_class4[0], NMLE_class4[1], X_test)]
    prediction = (prediction_pr == prediction_pr.max(axis=1)[:, None]).astype(int)

    confusion = np.matmul(prediction.transpose(), Y_test)
    print("\nConfusion Matrix on Test Data    :")
    print(confusion)
    print("\nAccuracy on Test :" + (confusion.trace() / Y_test.shape[0]).__str__())


Part_5()


