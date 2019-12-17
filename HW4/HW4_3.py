import cvxopt as co
import numpy as np
import scipy.io
np.set_printoptions(threshold=np.nan)


def Ker(x_i,x_j):
    dif=x_i-x_j
    return np.asscalar(np.exp(-0.0006*((np.matmul(dif,dif.transpose())))))




tmp = scipy.io.loadmat("dataset_4/mnist_train.mat")
x_train =np.array(tmp['X'])
y_train =np.array(tmp['Y']).reshape(-1,1)

yy=np.zeros(y_train.shape[0]).reshape(-1,1)
yy[np.where(y_train<=4)]=-1
yy[np.where(y_train>4)]=1
y_train=yy


tmp_test = scipy.io.loadmat("dataset_4/mnist_test.mat")
x_test =np.array(tmp_test['X'])
y_test =np.array(tmp_test['Y']).reshape(-1,1)

yy=np.zeros(y_test.shape[0]).reshape(-1,1)
yy[np.where(y_test<=4)]=-1
yy[np.where(y_test>4)]=1
y_test=yy


C_plus=1
C_neg=1







pp= np.zeros((x_train.shape[0],x_train.shape[0]))


for i in range(x_train.shape[0]):
    for j in range(x_train.shape[0]):
        pp[i,j]= Ker(x_train[i:i+1,:],x_train[j:j+1,:])
print(x_train.shape)
print(pp.shape)

P=np.multiply( pp   ,   np.matmul(y_train,y_train.transpose())  )
P=co.matrix(P.astype(np.double))

q= -1*np.ones(x_train.shape[0])
q=co.matrix(q.astype(np.double))

A=y_train.transpose()
A=co.matrix(A.astype(np.double))

b=np.array([0])
b=co.matrix(b.astype(np.double))

G=np.r_[np.identity(x_train.shape[0]),-1*np.identity(x_train.shape[0])]
G=co.matrix(G.astype(np.double))

rofo=np.zeros(y_train.shape[0])
rofo[np.where(y_train[:,0]==1)]=C_plus
rofo[np.where(y_train[:,0]==-1)]=C_neg
h=np.r_[rofo,np.zeros(x_train.shape[0])]
h=co.matrix(h.astype(np.double))


res=co.solvers.qp(P=P,q=q,A=A,b=b,G=G,h=h)
sol = np.array( res['x']).reshape(-1,1)
print("#############SOL#################")
print(np.sort(sol.reshape(-1)))

W_1=np.multiply(sol,y_train)

jojo= np.where(np.logical_and(sol>0.1,sol<0.9))[0]

print((y_train - np.matmul(pp,W_1)).shape)


W_0=  np.average( (y_train - np.matmul(pp,W_1))[jojo])


qq= np.zeros((x_test.shape[0],x_train.shape[0]))
for i in range(x_test.shape[0]):
    for j in range(x_train.shape[0]):
        qq[i,j]= Ker(x_test[i:i+1,:],x_train[j:j+1,:])


pred = np.sign(np.matmul(qq,W_1)+W_0)


####################


predd =  np.zeros((pred.shape[0],2))
y_testt= np.zeros((y_test.shape[0],2))

predd[    np.where(pred==1) [0],1]=1
predd[    np.where(pred==-1)[0],0]=1

y_testt[    np.where(y_test==1) [0],1]=1
y_testt[    np.where(y_test==-1)[0],0]=1



confusion_matrix=np.matmul(predd.transpose(),y_testt)



precision_N = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
recal_N     = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

precision_P = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
recal_P     = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

accuracy =  (confusion_matrix.trace())/(confusion_matrix.sum())


print("Confusion_matrix:")
print(confusion_matrix)


print("Precision Negative     "+precision_N.__str__())
print("Recall    Negative     "+recal_N.__str__())

print("Precision Positive     "+precision_P.__str__())
print("Recall    Positive     "+recal_P.__str__())

print("Accuracy :             "+accuracy.__str__())

