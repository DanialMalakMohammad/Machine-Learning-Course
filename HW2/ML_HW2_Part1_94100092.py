import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)


def get_data(name):

    data = (pd.read_csv(name)).values

    data_x = np.c_[np.copy(data[:, 0:6]), np.copy(data[:, 5:6]), np.copy(data[:, 5:6]) , np.copy(data[:, 5:6]),np.ones((data.shape[0],1)) ]
    data_y = np.c_[data[:, 6:7]]

    (data_x[:, 5:6])[data_x[:, 5:6] == 'southwest'] = 1
    (data_x[:, 6:7])[data_x[:, 6:7] == 'southeast'] = 1
    (data_x[:, 7:8])[data_x[:, 7:8] == 'northwest'] = 1
    (data_x[:, 8:9])[data_x[:, 8:9] == 'northeast'] = 1

    data_x[data_x == 'southwest'] = 0
    data_x[data_x == 'southeast'] = 0
    data_x[data_x == 'northwest'] = 0
    data_x[data_x == 'northeast'] = 0

    data_x[data_x == 'yes'] = 1
    data_x[data_x == 'no'] = 0

    data_x[data_x == 'male'] = 1
    data_x[data_x == 'female'] = 0

    data_x[:, 0:1] = np.power(data_x[:, 0:1], 2)

    data_x = data_x.astype(float)
    data_y = data_y.astype(float)

    return (data_x,data_y)

def linear_regression(xx,yy):
    return np.matmul(np.linalg.pinv(xx), yy)

def mean_square_loss(W,X,Y):
    error = np.subtract(Y,np.matmul(X,W))
    return np.asscalar(((np.matmul(error.transpose(),error))/Y.shape[0]))

def L2_linear_Regression(X,Y,landa):

    return linear_regression(   np.matmul(X.transpose(),X)+landa*np.identity(X.shape[1])      ,    np.matmul(X.transpose(),Y)     )

def five_fold_cross_validation_L2(X,Y,landa):

    x1  = X[200:1000,:]
    y1  = Y[200:1000,:]
    xx1 = X[0:200,:]
    yy1 = Y[0:200,:]

    x2  = np.r_[   X[0:200,:]   ,   X[400:1000,:]   ]
    y2  = np.r_[   Y[0:200,:]   ,   Y[400:1000,:]   ]
    xx2 =          X[200:400,:]
    yy2 =          Y[200:400,:]

    x3 = np.r_[    X[0:400,:]   ,   X[600:1000,:]]
    y3 = np.r_[    Y[0:400,:]   ,   Y[600:1000,:]]
    xx3 =          X[400:600,:]
    yy3 =          Y[400:600,:]

    x4 = np.r_[    X[0:600,:]   ,   X[800:1000,:]]
    y4 = np.r_[    Y[0:600,:]   ,   Y[800:1000,:]]
    xx4 =          X[600:800,:]
    yy4 =          Y[600:800,:]

    x5  =   X[0:800,:]
    y5  =   Y[0:800,:]
    xx5 =   X[800:1000,:]
    yy5 =   Y[800:1000,:]

    return (mean_square_loss(L2_linear_Regression(x1, y1,landa), xx1, yy1)+mean_square_loss(L2_linear_Regression(x2, y2,landa), xx2, yy2)+mean_square_loss(L2_linear_Regression(x3, y3,landa), xx3, yy3)+mean_square_loss(L2_linear_Regression(x4, y4,landa), xx4, yy4)+mean_square_loss(L2_linear_Regression(x5, y5,landa), xx5, yy5))/5





(x_train , y_train )=get_data("train.csv")
(x_test  , y_test  )=get_data("test.csv")






def Part6_1_1():
    print()
    print("***  Part 6.1.1   ****")
    print()

    w=linear_regression(x_train,y_train)
    print("W = "+w.__str__())

    print("Mean Square Loss on Train :  "+mean_square_loss(w,x_train,y_train).__str__())
    print("Mean Square Loss on Test  :  "+mean_square_loss(w,x_test,y_test).__str__())
    print("#######################")

def Part6_1_2():

    print()
    print("***  Part 6.1.2   ****")
    print()

    W=np.ones((10,110))

    error_train = np.ones(100)
    error_whole_train = np.ones(100)
    error_test= np.ones(100)



    for i in range(100):
        W[:,i:i+1]=linear_regression(x_train[0:10+i*10,:],y_train[0:10+i*10,:])

        error_train[i]=mean_square_loss(W[:,i:i+1],x_train[0:10+i*10,:],y_train[0:10+i*10,:])
        error_whole_train[i]=mean_square_loss(W[:,i:i+1],x_train,y_train)
        error_test[i]=mean_square_loss(W[:,i:i+1],x_test,y_test)


       # print(error_scaler[i])
       # print("######")

    plt.plot(np.arange(100), error_train,       label="error_train")
    plt.plot(np.arange(100), error_whole_train, label="error_whole_train")
    plt.plot(np.arange(100), error_test,        label="error_test")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def Part6_2_1_Calculate():
    ## Part 6.2.1

    W_gradient = np.ones((10, 1))
    etah = 0.0000000003

    AA = np.matmul(x_train.transpose(), y_train)
    BB = np.matmul(x_train.transpose(), x_train)

    W_BGD = np.zeros((10, 10000))
    MSE_BDG = np.zeros((1, 10000))

    W_BGD[:, 0:1] = W_gradient;
    MSE_BDG[0, 0] = mean_square_loss(W_gradient, x_train, y_train)

    for i in range(1000000000):

        if ((i % 1000000) == 0):
            np.savetxt("W_BGD.txt", W_BGD)
            np.savetxt("MSE_BGD.txt", MSE_BDG)

        # proob=np.matmul(x_train.transpose(),np.subtract(y_train,np.matmul(x_train,W_gradient)))
        proob = np.subtract(AA, np.matmul(BB, W_gradient))
        W_gradient = W_gradient + etah * proob

        if ((i % 100000) == 99999):
            tt = (int)(i / 100000) + 1
            W_BGD[:, tt:tt + 1] = W_gradient;
            MSE_BDG[0, tt] = mean_square_loss(W_gradient, x_train, y_train)
            print((i + 1).__str__() + " : " + MSE_BDG[0, tt].__str__())

    np.savetxt("W_BGD.txt", W_BGD)
    np.savetxt("MSE_BGD.txt", MSE_BDG)

    exit(0)

def Part6_2_1_Show_Results():

    print()
    print("   Part6_2_2   ")
    print()

    w = np.loadtxt("W_BGD.txt")[:, 7430].reshape(10, 1)
    mse = np.loadtxt("MSE_BGD.txt")

    plt.plot(np.arange(7430), mse[0:7430], label="error_train")
    plt.show()

    # print(np.c_[linear_regression(x_train,y_train),w])
    print("Batch Gradient Descend MSE on Train  :   " + mean_square_loss(w, x_train, y_train).__str__())
    print("Closed Form MSE on Train             :   " + mean_square_loss(linear_regression(x_train, y_train), x_train,
                                                                         y_train).__str__())

    print()
    print()
    print("Batch Gradient Descend MSE on Test  :   " + mean_square_loss(w, x_test, y_test).__str__())
    print("Closed Form MSE on Test             :   " + mean_square_loss(linear_regression(x_train, y_train), x_test,
                                                                        y_test).__str__())

    print("W_BGD    :")

    print(w)

    exit(0)


def Part6_2_2_Calculate():

    W_gradient = np.zeros((10, 1))
    etah = 0.0000002000

    AA = np.matmul(x_train.transpose(), y_train)
    BB = np.matmul(x_train.transpose(), x_train)

    W_SGD = np.zeros((10, 10000))
    MSE_SGD = np.zeros((1, 10000))

    W_SGD[:, 0:1] = W_gradient;
    MSE_SGD[0, 0] = mean_square_loss(W_gradient, x_train, y_train)

    sample=0

    for i in range(1000000000):

        if ((i % 1000000) == 0):
            np.savetxt("W_SGD.txt", W_SGD)
            np.savetxt("MSE_SGD.txt", MSE_SGD)
            etah=etah*0.998


        W_gradient = W_gradient + etah * (np.asscalar(y_train[sample])  - np.asscalar( np.matmul(x_train[sample:sample+1,:],W_gradient))  )*x_train[sample:sample+1,:].transpose()

        if ((i % 100000) == 99999):
            tt = (int)(i / 100000) + 1
            W_SGD[:, tt:tt + 1] = W_gradient;
            MSE_SGD[0, tt] = mean_square_loss(W_gradient, x_train, y_train)
            print((i + 1).__str__() + " : " + MSE_SGD[0, tt].__str__())
        sample=(sample+1)%1000

def Part6_2_2_Show_Results():

    print()
    print("   Part6_2_2   ")
    print()

    w = np.loadtxt("W_SGD.txt")[:, 9990].reshape(10, 1)
    mse = np.loadtxt("MSE_SGD.txt")

    plt.plot(np.arange(9990), mse[0:9990])
    plt.show()

    # print(np.c_[linear_regression(x_train,y_train),w])
    print("Sample Mode Gradient Descend MSE on Train  :   " + mean_square_loss(w, x_train, y_train).__str__())
    print("Closed Form MSE on Train             :   " + mean_square_loss(linear_regression(x_train, y_train), x_train,
                                                                         y_train).__str__())

    print()
    print()
    print("Sample Mode Gradient Descend MSE on Test  :   " + mean_square_loss(w, x_test, y_test).__str__())
    print("Closed Form MSE on Test             :   " + mean_square_loss(linear_regression(x_train, y_train), x_test,
                                                                        y_test).__str__())

    print("W_SGD    :")

    for i in range(10):
        print(np.asscalar(w[i]))


    exit(0)


def Part6_3():

    print()
    print("*****Part 6_3 : *****")
    print()
    print("5 fold cross validation for different L2 regularization coeficents :")

    for i in range(-4,5):
        print("(10)^"+i.__str__()+" : "+five_fold_cross_validation_L2(x_train,y_train,10**i).__str__())

    print("Thus we choose landa= (10)^(-1) and train the model with whole train_data(train and validation data). Then we calculated its MSE on train and test data")

    ww= L2_linear_Regression(x_train,y_train,(10)**(-1))
    print("MSE on whole train :  "+mean_square_loss(ww,x_train,y_train).__str__())
    print("MSE on test        :  "+mean_square_loss(ww,x_test,y_test  ).__str__())


