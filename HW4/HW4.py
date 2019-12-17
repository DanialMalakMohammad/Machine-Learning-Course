import cvxopt as co
import numpy as np
import scipy.io
np.set_printoptions(threshold=np.nan)





tmp = scipy.io.loadmat("dataset_4/scene_train.mat")
x_train =np.array(tmp['X'])
y_train =np.array(tmp['Y']).reshape(-1,1)

tmp_test = scipy.io.loadmat("dataset_4/scene_test.mat")
x_test =np.array(tmp_test['X'])
y_test =np.array(tmp_test['Y']).reshape(-1,1)




C_neg=list(y_train).count(1)/y_train.shape[0]
C_plus=list(y_train).count(-1)/y_train.shape[0]

#C_plus=1
#C_neg=1





P=np.multiply( np.matmul( x_train , x_train.transpose())   ,   np.matmul(y_train,y_train.transpose())  )
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

W_1=np.matmul(x_train.transpose(),np.multiply(sol,y_train))

jojo= np.where(np.logical_and(sol>0.01,sol<0.05))[0]

W_0= np.average( y_train[jojo]-np.matmul(x_train,W_1)[jojo] )




pred=np.sign(np.matmul(x_test,W_1)+W_0)



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


print("Confusion_matrix:")
print(confusion_matrix)


print("Precision Negative     "+precision_N.__str__())
print("Recall    Negative     "+recal_N.__str__())

print("Precision Positive     "+precision_P.__str__())
print("Recall    Positive     "+recal_P.__str__())




