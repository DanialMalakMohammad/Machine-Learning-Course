import numpy as np
import  scipy.io
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
lr=0.001
epoch=10000000

tmp = scipy.io.loadmat("dataset_4/mnist_train.mat")
x_train =np.array(tmp['X'])
y_train =np.array(tmp['Y']).reshape(-1,1)

tmp_test = scipy.io.loadmat("dataset_4/mnist_test.mat")
x_test =np.array(tmp_test['X'])
y_test =np.array(tmp_test['Y']).reshape(-1,1)

YY = np.zeros((y_train.shape[0], 10))
YY[np.arange(y_train.shape[0]), (y_train).reshape(-1)] = 1
y_train = YY

YY = np.zeros((y_test.shape[0], 10))
YY[np.arange(y_test.shape[0]), (y_test).reshape(-1)] = 1
y_test = YY




def forward_fc(W,X,B):
    return np.maximum( np.matmul(X, W) + B.repeat(repeats=X.shape[0], axis=0) , np.zeros((X.shape[0], B.shape[1]))  )

def backward_fc(E_i,W,X,B):
    #W[d,i,j] = E_i[d,j] * f[d,j] * x_i[d,i]
    # XW=O --> x'[j]= W[i,j] x[i]
    #E_o[d,j]= sigma W[j,t]*B[d,t]
    #B[d,t]*WW[t,j]
    B_g = np.multiply(E_i, np.sign(forward_fc(W, X, B)))
    W_g = (X.reshape((X.shape[0], X.shape[1], 1))) * (B_g.reshape((B_g.shape[0], 1, B_g.shape[1])))
    E_o = np.matmul(B_g, W.transpose())
    return (E_o,W_g,B_g)


def forward_sm(X):
    return np.exp(X) /  np.sum(np.exp(X),axis=1).reshape((-1,1))

def backward_sm(X,Y):
    return  forward_sm(X)-Y




W_1=0.1*np.random.rand(x_train.shape[1],32)
W_2=0.1*np.random.rand(32,32)
W_3=0.1*np.random.rand(32,32)
W_4=0.1*np.random.rand(32,10)

B_1=0.1*np.random.rand(1,32)
B_2=0.1*np.random.rand(1,32)
B_3=0.1*np.random.rand(1,32)
B_4=0.1*np.random.rand(1,10)


loss_train=[]
loss_test=[]
accuracy_train=[]
accuracy_test=[]

for ii in range(epoch):

    #Forwarding

    X_1=forward_fc(W_1,x_train,B_1)
    X_2=forward_fc(W_2,X_1,B_2)
    X_3=forward_fc(W_3,X_2,B_3)
    X_4=forward_fc(W_4,X_3,B_4)

    XX_1=forward_fc(W_1,x_test,B_1)
    XX_2=forward_fc(W_2,XX_1,B_2)
    XX_3=forward_fc(W_3,XX_2,B_3)
    XX_4=forward_fc(W_4,XX_3,B_4)

    if(ii%100==0):
        predict_train=forward_sm(X_4)
        predict_test=forward_sm(XX_4)
        loss_train.append(-1*np.average(np.log( np.max(np.multiply(predict_train,y_train),axis=1))))
        loss_test.append(-1 * np.average(np.log(np.max(np.multiply(predict_test, y_test), axis=1))))
        accuracy_train.append(np.average(np.equal(np.argmax(predict_train,axis=1),np.argmax(y_train,axis=1))))
        accuracy_test.append(np.average(np.equal(np.argmax(predict_test, axis=1), np.argmax(y_test, axis=1))))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Epoch              :   "+ii.__str__())
        print("Loss     on Train  :   "+loss_train[-1].__str__())
        print("Loss     on Test   :   "+loss_test[-1].__str__())
        print("Accuracy on Train  :   "+accuracy_train[-1].__str__())
        print("Accuracy on Test   :   "+accuracy_test[-1].__str__())

    if(ii%3000==0):
        plt.clf()
        plt.figure(1)
        plt.plot(100 * np.arange(len(loss_train)), loss_train, label='train')
        plt.plot(100 * np.arange(len(loss_test)), loss_test, label='test')
        plt.title('Mnist Neural Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Cross_entropy Loss')
        plt.legend()
        plt.savefig('mnist_Loss_'+ii.__str__()+'.png',dpi=1200)
        plt.clf()
        plt.plot(100 * np.arange(len(accuracy_train)), accuracy_train, label='train')
        plt.plot(100 * np.arange(len(accuracy_test)), accuracy_test, label='test')
        plt.title('Mnist Neural Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('mnist_Accuracy_'+ii.__str__()+'.png',dpi=1200)

    #Backprop

    D_4=backward_sm(X_4,y_train)
    (D_3, W_g_4, B_g_4) =   backward_fc(D_4,W_4,X_3,B_4)
    (D_2, W_g_3, B_g_3) =   backward_fc(D_3,W_3,X_2,B_3)
    (D_1, W_g_2, B_g_2) =   backward_fc(D_2,W_2,X_1,B_2)
    (D_0, W_g_1, B_g_1) =   backward_fc(D_1,W_1,x_train,B_1)

    W_1=W_1-lr*(np.average(W_g_1,axis=0).reshape(W_1.shape))
    W_2=W_2-lr*(np.average(W_g_2,axis=0).reshape(W_2.shape))
    W_3=W_3-lr*(np.average(W_g_3,axis=0).reshape(W_3.shape))
    W_4=W_4-lr*(np.average(W_g_4,axis=0).reshape(W_4.shape))

    B_1 = B_1 - lr * np.average(B_g_1,axis=0)
    B_2 = B_2 - lr * np.average(B_g_2,axis=0)
    B_3 = B_3 - lr * np.average(B_g_3,axis=0)
    B_4 = B_4 - lr * np.average(B_g_4,axis=0)





