

import numpy as np
import pickle
import sklearn.cluster
import matplotlib.pyplot as plt


#Accuracies= [0.9230769230769231, 0.9358974358974359, 0.9423076923076923, 0.9665809768637532, 0.9678249678249679]
#Accuracies =[0.8974358974358975, 0.9102564102564102, 0.9166666666666666, 0.9665809768637532, 0.9626769626769627]
#Accuracies=[0.8461538461538461, 0.9358974358974359, 0.9487179487179487, 0.9691516709511568, 0.9703989703989704]
#Accuracies=[0.8205128205128205, 0.9230769230769231, 0.9358974358974359, 0.9511568123393316, 0.9691119691119691]
#Accuracies = [0.8205128205128205, 0.8974358974358975, 0.9294871794871795, 0.961439588688946, 0.9626769626769627]


#clf='Logistic Regression : '
#clf ='Random Forest '
#clf = 'SVM'
#clf = 'Adaboost'
clf = 'Neural Network'



Accuracies=np.array(Accuracies)
ratios=[5,10,20,50,100]

ss = np.arange(5)

plt.xlabel('Percentage of Data for Learning')
plt.title(clf +', Accuracy / percentage of Data used for learning')
plt.ylabel('Accuracy')
plt.plot(ss,Accuracies,'go-')
plt.xticks(ss,ratios)
plt.show()