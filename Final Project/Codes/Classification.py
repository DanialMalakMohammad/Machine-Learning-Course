import numpy as np
import sklearn.model_selection, sklearn.linear_model, sklearn.ensemble, sklearn.svm, sklearn.tree, sklearn.feature_extraction,sklearn.cluster,sklearn.mixture , sklearn.decomposition
import pickle
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.layers import BatchNormalization, Dropout, Dense
import keras.optimizers

np.set_printoptions(threshold=np.inf)



x_train = []
y_train = []
subject_id_train = []
x_train = np.loadtxt("ML_Proj_data - v1/Train/X_train.txt")
y_train = np.loadtxt("ML_Proj_data - v1/Train/y_train.txt")
subject_id_train = np.loadtxt("ML_Proj_data - v1/Train/subject_id_train.txt")



def encode(a):
    a=a.astype(int)
    soso = np.zeros((a.shape[0], 12))
    soso[np.arange(a.shape[0]), a] = 1
    return soso

def get_NN(l1=1024,l2=256,dropout1=0,dropout2=0,dropout3=0.25,batch_size=128,epochs=250,ll=0,regularization='l2'):

    andex = np.random.choice(x_train.shape[0], int(x_train.shape[0]), replace=False)

    xnn_train = (x_train[andex])[0:int(x_train.shape[0] * 0.9)]
    ynn_train = (y_train[andex])[0:int(y_train.shape[0] * 0.9)]-1
    xnn_val = (x_train[andex])[int(x_train.shape[0] * 0.9):]
    ynn_val = (y_train[andex])[int(y_train.shape[0] * 0.9):]-1
    ynn_train = encode(ynn_train)
    mat = np.copy(ynn_val)
    ynn_val = encode(ynn_val)



    """
    inputs = Input(shape=(561,), name="inputs")
    inputs_0 = Dropout(dropout1, name='dropout_1')(inputs)
    dense_1 = Dense(l1, activation="relu", name="inputs_1")(inputs_0)
    #dense_1 = BatchNormalization(name="inputs_2")(dense_1)
    dense_1 = Dropout(dropout2, name='dropout_2')(dense_1)
    dense_2 = Dense(l2, activation="relu", name="inputs_3")(dense_1)
    #dense_2 = BatchNormalization(name="inputs_4")(dense_2)
    dense_2 = Dropout(dropout3, name='dropout_3')(dense_2)
    dense_3 = Dense(12, activation="relu")(dense_2)
    dense_3 = keras.layers.Softmax()(dense_3)
    """

    if(regularization=='l2'):

        inputs = Input(shape=(561,), name="inputs")
        inputs_0 = Dropout(dropout1, name='dropout_1')(inputs)
        dense_1 = Dense(l1, activation="relu", name="inputs_1",kernel_regularizer=keras.regularizers.l2(ll),activity_regularizer=keras.regularizers.l2(ll),bias_regularizer=keras.regularizers.l2(ll))(inputs_0)
        #dense_1 = BatchNormalization(name="inputs_2")(dense_1)
        dense_1 = Dropout(dropout2, name='dropout_2')(dense_1)
        dense_2 = Dense(l2, activation="relu", name="inputs_3",kernel_regularizer=keras.regularizers.l2(ll),activity_regularizer=keras.regularizers.l2(ll),bias_regularizer=keras.regularizers.l2(ll))(dense_1)
        #dense_2 = BatchNormalization(name="inputs_4")(dense_2)
        dense_2 = Dropout(dropout3, name='dropout_3')(dense_2)
        dense_3 = Dense(12, activation="relu",kernel_regularizer=keras.regularizers.l2(ll),activity_regularizer=keras.regularizers.l2(ll),bias_regularizer=keras.regularizers.l2(ll))(dense_2)
        dense_3 = keras.layers.Softmax()(dense_3)

    elif(regularization=='l1'):

        inputs = Input(shape=(561,), name="inputs")
        inputs_0 = Dropout(dropout1, name='dropout_1')(inputs)
        dense_1 = Dense(l1, activation="relu", name="inputs_1", kernel_regularizer=keras.regularizers.l1(ll),activity_regularizer=keras.regularizers.l1(ll), bias_regularizer=keras.regularizers.l1(ll))(inputs_0)
        # dense_1 = BatchNormalization(name="inputs_2")(dense_1)
        dense_1 = Dropout(dropout2, name='dropout_2')(dense_1)
        dense_2 = Dense(l2, activation="relu", name="inputs_3", kernel_regularizer=keras.regularizers.l1(ll),activity_regularizer=keras.regularizers.l1(ll), bias_regularizer=keras.regularizers.l1(ll))(dense_1)
        # dense_2 = BatchNormalization(name="inputs_4")(dense_2)
        dense_2 = Dropout(dropout3, name='dropout_3')(dense_2)
        dense_3 = Dense(12, activation="relu", kernel_regularizer=keras.regularizers.l1(ll),activity_regularizer=keras.regularizers.l1(ll), bias_regularizer=keras.regularizers.l1(ll))(dense_2)
        dense_3 = keras.layers.Softmax()(dense_3)


    model = Model(inputs=inputs, outputs=dense_3)

    model.compile(optimizer='adam', loss="categorical_crossentropy")




    model.fit(x=xnn_train,  # load x_train data
              y=ynn_train,  # load y_train data
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2
              )


    pr=np.argmax(model.predict(xnn_val),axis=1)
    acc=len((np.where(pr-mat==0))[0])/pr.shape[0]
    print("accuracies = ",acc)



    return model,acc

def get_NN2(x_train,y_train,l1=1024,l2=256,dropout1=0,dropout2=0,dropout3=0.25,batch_size=128,epochs=250):

    andex = np.random.choice(x_train.shape[0], int(x_train.shape[0]), replace=False)

    xnn_train = (x_train[andex])[0:int(x_train.shape[0] * 0.9)]
    ynn_train = (y_train[andex])[0:int(y_train.shape[0] * 0.9)]-1
    xnn_val = (x_train[andex])[int(x_train.shape[0] * 0.9):]
    ynn_val = (y_train[andex])[int(y_train.shape[0] * 0.9):]-1
    ynn_train = encode(ynn_train)
    mat = np.copy(ynn_val)
    ynn_val = encode(ynn_val)




    inputs = Input(shape=(561,), name="inputs")
    inputs_0 = Dropout(dropout1, name='dropout_1')(inputs)
    dense_1 = Dense(l1, activation="relu", name="inputs_1")(inputs_0)
    #dense_1 = BatchNormalization(name="inputs_2")(dense_1)
    dense_1 = Dropout(dropout2, name='dropout_2')(dense_1)
    dense_2 = Dense(l2, activation="relu", name="inputs_3")(dense_1)
    #dense_2 = BatchNormalization(name="inputs_4")(dense_2)
    dense_2 = Dropout(dropout3, name='dropout_3')(dense_2)
    dense_3 = Dense(12, activation="relu")(dense_2)
    dense_3 = keras.layers.Softmax()(dense_3)




    model = Model(inputs=inputs, outputs=dense_3)

    model.compile(optimizer='adam', loss="categorical_crossentropy")




    model.fit(x=xnn_train,  # load x_train data
              y=ynn_train,  # load y_train data
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2
              )


    pr=np.argmax(model.predict(xnn_val),axis=1)
    acc=len((np.where(pr-mat==0))[0])/pr.shape[0]
    print("accuracies = ",acc)



    return model,acc


def Part1_logistic_regression():
    clf_logistic_regression = sklearn.linear_model.LogisticRegression(fit_intercept=True, intercept_scaling=1,
                                                                      class_weight=None, random_state=None,
                                                                      solver='newton-cg', max_iter=100,
                                                                      multi_class='multinomial')

    print(np.array(
        sklearn.model_selection.cross_val_score(estimator=clf_logistic_regression, X=x_train, y=y_train, cv=3)).mean())

def Part1_random_forest():
    clf_random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None,
                                                                min_samples_split=2, min_samples_leaf=1,
                                                                min_weight_fraction_leaf=0, max_features='auto',
                                                                max_leaf_nodes=None, min_impurity_decrease=0,
                                                                bootstrap=True, oob_score=False, random_state=None,
                                                                warm_start=False, class_weight=None)
    # 0.957129349214285

    print(np.array(
        sklearn.model_selection.cross_val_score(estimator=clf_random_forest, X=x_train, y=y_train, cv=3)).mean())

def Part1_svm():
    clf_svm = sklearn.svm.SVC(C=100, kernel='rbf', degree=1, gamma='auto', coef0=1, shrinking=True, probability=False,
                              tol=1e-5, class_weight=None, decision_function_shape='ovo', random_state=None)
    # clf_svm = sklearn.svm.SVC(C=100,kernel='poly',degree=20,gamma='auto',coef0=1,shrinking=True,probability=False,tol=1e-5,class_weight=None,decision_function_shape='ovo',random_state=None)

    print(np.array(sklearn.model_selection.cross_val_score(estimator=clf_svm, X=x_train, y=y_train, cv=3)).mean())

def Part1_adaboost():
    clf_adaboost = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=7),
                                                       n_estimators=50, learning_rate=1, algorithm='SAMME',
                                                       random_state=None)

    print(np.array(sklearn.model_selection.cross_val_score(estimator=clf_adaboost, X=x_train, y=y_train, cv=3)).mean())




###################################################

penalties = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.1, 1, 10, 100]


def Part1_2_LR_L2():
    print("Logistic Regression , L2")
    print()

    accuracy = []
    non_zero_coef_ratio = []
    non_zero_coef_mean = []

    for penalty in penalties:
        clf_logistic_regression = sklearn.linear_model.LogisticRegression(penalty='l2', C=1 / penalty,
                                                                          random_state=None,
                                                                          solver='saga', multi_class='multinomial')

        accuracy.append(np.array(
            sklearn.model_selection.cross_val_score(estimator=clf_logistic_regression, X=x_train, y=y_train,
                                                    cv=3)).mean())

        clf_logistic_regression = sklearn.linear_model.LogisticRegression(penalty='l2', C=1 / penalty,
                                                                          random_state=None, solver='saga',
                                                                          multi_class='multinomial')

        clf_logistic_regression.fit(x_train, y_train)

        non_zero_coef_ratio.append( np.argwhere(np.abs(clf_logistic_regression.coef_)>0.000001).shape[0]/(clf_logistic_regression.coef_.shape[0]*clf_logistic_regression.coef_.shape[1]))
        non_zero_coef_mean.append((clf_logistic_regression.coef_[(np.argwhere(np.abs(clf_logistic_regression.coef_)>0.000001))[0]]).mean())
    print("Accuracies          :    ", accuracy)
    print("non_zero_coef_ratio :    ", non_zero_coef_ratio)
    print('non_zero_coef_mean  :    ',non_zero_coef_mean)






    #Accuracies          =     [0.9675566096268594, 0.9679424122108881, 0.9675561618865154, 0.9676851608545237, 0.9675561618865154, 0.9678141598225319, 0.9678141598225319, 0.9678134132428798, 0.9675563107875513, 0.9615067997993748, 0.926864763544548]
    #non_zero_coef_ratio =     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    #non_zero_coef_mean  =     [0.028698143777518795, 0.028703294137539782, 0.029120929561890108, 0.02845386026340359, 0.029628570885337283, 0.028529983139965104, 0.028461755416071582, 0.028033630788580906, 0.023406771646678894, 0.013650313611866902, 0.007965741430481468]

def Part1_2_LR_L1():
    print("Logistic Regression , L1")
    print()

    accuracy = []
    non_zero_coef_ratio = []
    non_zero_coef_mean=[]
    for penalty in penalties:
        clf_logistic_regression = sklearn.linear_model.LogisticRegression(penalty='l1', C=1/penalty, random_state=None,
                                                                          solver='saga', multi_class='multinomial')

        accuracy.append(np.array(
            sklearn.model_selection.cross_val_score(estimator=clf_logistic_regression, X=x_train, y=y_train,
                                                    cv=3)).mean())

        clf_logistic_regression = sklearn.linear_model.LogisticRegression(penalty='l1', C=1/penalty, random_state=None,
                                                                          solver='saga', multi_class='multinomial')

        clf_logistic_regression.fit(x_train, y_train)

        non_zero_coef_ratio.append( np.argwhere(np.abs(clf_logistic_regression.coef_)>0.000001).shape[0]/(clf_logistic_regression.coef_.shape[0]*clf_logistic_regression.coef_.shape[1]))
        non_zero_coef_mean.append((clf_logistic_regression.coef_[(np.argwhere(np.abs(clf_logistic_regression.coef_)>0.000001))[0]]).mean())

    print("Accuracies          :    ", accuracy)
    print("non_zero_coef_ratio :    ", non_zero_coef_ratio)
    print('non_zero_coef_mean  :    ',non_zero_coef_mean)

    #Accuracies          :     [0.9688427175076454, 0.9687141662799812, 0.9689718653766896, 0.9689717164756536, 0.9689715665373814, 0.9690999688640097, 0.9692294155723621, 0.9698735149317153, 0.967297866148427, 0.933949401496435, 0.8057092207971451]
    #non_zero_coef_ratio :     [1.0, 1.0, 1.0, 1.0, 1.0, 0.9998514557338086, 0.9985145573380867, 0.8207070707070707, 0.24420677361853832, 0.03832442067736185, 0.006833036244800951]
    # non_zero_coef_count=[561, 561, 561, 561, 561, 560, 560, 460, 137, 21, 3]

def Part1_2_LinearSVM_L2():
    print("Linear SVM , L2")
    print()

    accuracy = []
    non_zero_coef_ratio = []
    non_zero_coef_mean = []

    for penalty in penalties:
        clf_svm = sklearn.svm.LinearSVC(penalty='l2', loss='hinge', dual=True, C=1/penalty, random_state=None)

        accuracy.append(np.array(
            sklearn.model_selection.cross_val_score(estimator=clf_svm, X=x_train, y=y_train,
                                                    cv=3)).mean())

        clf_svm = sklearn.svm.LinearSVC(penalty='l2', loss='hinge', dual=True, C=1/penalty, random_state=None)

        clf_svm.fit(x_train, y_train)

        non_zero_coef_ratio.append( np.argwhere(np.abs(clf_svm.coef_)>0.000001).shape[0]/(clf_svm.coef_.shape[0]*clf_svm.coef_.shape[1]))
        non_zero_coef_mean.append((clf_svm.coef_[(np.argwhere(np.abs(clf_svm.coef_)>0.000001))[0]]).mean())

    print("Accuracies          :    ", accuracy)
    print("non_zero_coef_ratio :    ", non_zero_coef_ratio)
    print('non_zero_coef_mean  :    ',non_zero_coef_mean)

    # Accuracies =   [0.9680743964602687, 0.9665295451010505, 0.9692298643499423, 0.9684563193194732, 0.9424490015450258, 0.9221012146478303, 0.886821195385722, 0.8672560593013849, 0.8246414665892744, 0.7905296008475085, 0.7700540041161195]
    # non_zero_coef_ratio =   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9998514557338086, 0.9998514557338086, 0.9994058229352347, 0.9998514557338086]
    #non_zero_coef_mean  =     [0.016628287878465042, 0.016749782795681924, 0.01674911054322412, 0.01676298138018625, 0.016424436851222395, 0.016763745108559512, 0.016762480215477736, 0.01682794242597516, 0.016370539715996776, 0.013041892663367386, 0.011093211560926004]

def Part1_2_LinearSVM_L1():
    print("Linear SVM , L1")
    print()

    accuracy = []
    non_zero_coef_ratio = []
    non_zero_coef_mean=[]

    for penalty in penalties:
        clf_svm = sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=1/penalty, random_state=None)

        accuracy.append(
             np.array(sklearn.model_selection.cross_val_score(estimator=clf_svm, X=x_train, y=y_train, cv=3)).mean())

        clf_svm = sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=1/penalty, random_state=None)

        clf_svm.fit(x_train, y_train)


        non_zero_coef_ratio.append( np.argwhere(np.abs(clf_svm.coef_)>0.000001).shape[0]/(clf_svm.coef_.shape[0]*clf_svm.coef_.shape[1]))
        non_zero_coef_mean.append((clf_svm.coef_[(np.argwhere(np.abs(clf_svm.coef_)>0.000001))[0]]).mean())


    print("Accuracies          :    ", accuracy)
    print("Non_zero_coef_ratio :    ", non_zero_coef_ratio)
    print('non_zero_coef_mean  :    ',non_zero_coef_mean)

    # Accuracies =            [0.9639535974784542, 0.9654987487142169, 0.9696168622912035, 0.9630486658772207, 0.8874723107267801, 0.8637795054682508, 0.7930946486146304, 0.7546042920549958, 0.7059464453099258, 0.48152201926775406, 0.15450012974904354]
    # non_zero_coef_ratio =   [0.6081402257872847, 0.3530897207367796, 0.2043969102792632, 0.10308972073677956, 0.030897207367795602, 0.020202020202020204, 0.012477718360071301, 0.006981580510992276, 0.004010695187165776, 0.0016339869281045752, 0.00029708853238265005]
    # non_zero_coef_count =   [341, 198, 114, 57, 17, 11, 7, 3, 2, 0, 0]

def Part1_2_NN_L2():


    accuracies=[]
    non_zero_coef_ratio=[]
    non_zero_coef_mean=[]
    for penalty in penalties:

        (model,ac)=get_NN(l1=150, l2=50, dropout1=0.2, dropout2=0.2, dropout3=0.2, batch_size=128, epochs=250, ll=penalty, regularization='l2')


        rodo=model.get_weights()
        sod=np.array([1]).reshape(-1)
        for rod in rodo:
            sod = np.r_[np.copy(sod),np.copy(rod.reshape(-1))]



        sod=sod[1:]


        accuracies.append(ac)
        non_zero_coef_ratio.append(np.argwhere(np.abs(sod) > 0.000001).shape[0] / (sod.shape[0]))
        non_zero_coef_mean.append(sod[np.argwhere(np.abs(sod) > 0.000001)].mean())

    print('accuracies',accuracies)
    print('non_zero_coef_ratio',non_zero_coef_ratio)
    print('non_zero_coef_mean',non_zero_coef_mean)
    exit()

    #accuracies[0.9536679536679536, 0.924066924066924, 0.915057915057915, 0.31016731016731014, 0.17117117117117117, 0.17117117117117117, 0.1879021879021879, 0.15315315315315314, 0.15186615186615188, 0.14157014157014158, 0.1287001287001287]
    #non_zero_coef_ratio[0.5138218943998616, 0.43935887175272004, 0.26415175964179877, 0.0619930349765309, 0.023166273712444032, 0.027632973545889124, 0.018299409487140664, 0.024377582141852868, 0.04784668296164911, 0.0485604897146936, 0.0728299193182064]
    #non_zero_coef_mean[-0.0012592921300287013, -0.00020892376919545773, 0.0009705267563054048, 0.0011942332785754722, 0.0013276035922475116, 0.0009539807934903034, 0.002924665850188823, 0.0009494600083348451, 0.0015698027326537035, 0.001462276223901553, 0.0003998379810197039]

def Part1_2_NN_L1():


    accuracies=[]
    non_zero_coef_ratio=[]
    non_zero_coef_mean=[]
    for penalty in penalties:

        (model,ac)=get_NN(l1=150, l2=50, dropout1=0.2, dropout2=0.2, dropout3=0.2, batch_size=128, epochs=250, ll=penalty, regularization='l1')


        rodo=model.get_weights()
        sod=np.array([1]).reshape(-1)
        for rod in rodo:
            sod = np.r_[np.copy(sod),np.copy(rod.reshape(-1))]



        sod=sod[1:]


        accuracies.append(ac)
        non_zero_coef_ratio.append(np.argwhere(np.abs(sod) > 0.000001).shape[0] / (sod.shape[0]))
        non_zero_coef_mean.append(sod[np.argwhere(np.abs(sod) > 0.000001)].mean())

    print('accuracies',accuracies)
    print('non_zero_coef_ratio',non_zero_coef_ratio)
    print('non_zero_coef_mean',non_zero_coef_mean)
    exit()

    #accuracies=[0.9343629343629344, 0.5817245817245817, 0.21364221364221364, 0.1274131274131274, 0.15057915057915058, 0.14414414414414414, 0.15701415701415702, 0.15444015444015444, 0.17631917631917632, 0.15057915057915058, 0.15958815958815958]
    #non_zero_coef_ratio=[0.9980424390560446, 0.9977828729640285, 0.9976639051718544, 0.9977720577101945, 0.9974584153490083, 0.997079881464818, 0.9975557526335143, 0.9977396119486924, 0.9977287966948585, 0.9976530899180204, 0.9979126560100365]
    #non_zero_coef_mean[0.00026271417277812844, 0.001765048697260668, 0.0015733848615280363, 0.0017259649139436549, 0.0016063653026554229, 0.001431069294718449, 0.0015273715516848404, 0.001826709350747827, 0.0016575432997496136, 0.001499898687659678, 0.001964997056000199]



##########################################

ratios=[0.05,0.1,0.2,0.5,1]

def Part3_LR():

    accuracies=[]

    for ratio in ratios:


        clf = sklearn.linear_model.LogisticRegression(random_state=None, solver='newton-cg', multi_class='multinomial')

        print(ratio)
        indx = np.random.choice(x_train.shape[0], int(x_train.shape[0]*ratio) , replace=False)
        xx=x_train[indx]
        yy=y_train[indx]
        xx_train = xx[0:int(xx.shape[0]*0.9) ]
        yy_train = yy[0:int(yy.shape[0]*0.9) ]
        xx_val   = xx[int(xx.shape[0]*0.9):  ]
        yy_val   = yy[int(yy.shape[0] *0.9): ]
        clf.fit(xx_train,yy_train)
        accuracies.append(clf.score(xx_val,yy_val))
    print("Accuracies : ", accuracies)

    #Accuracies: [0.9230769230769231, 0.9358974358974359, 0.9423076923076923, 0.9665809768637532, 0.9678249678249679]

def Part3_RF():
    accuracies = []

    for ratio in ratios:

        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy',
                                                                    max_depth=None,
                                                                    min_samples_split=2, min_samples_leaf=1,
                                                                    min_weight_fraction_leaf=0, max_features='auto',
                                                                    max_leaf_nodes=None, min_impurity_decrease=0,
                                                                    bootstrap=True, oob_score=False, random_state=None,
                                                                    warm_start=False, class_weight=None)

        print(ratio)
        indx = np.random.choice(x_train.shape[0], int(x_train.shape[0] * ratio), replace=False)
        xx = x_train[indx]
        yy = y_train[indx]
        xx_train = xx[0:int(xx.shape[0] * 0.9)]
        yy_train = yy[0:int(yy.shape[0] * 0.9)]
        xx_val = xx[int(xx.shape[0] * 0.9):]
        yy_val = yy[int(yy.shape[0] * 0.9):]
        clf.fit(xx_train, yy_train)
        accuracies.append(clf.score(xx_val, yy_val))
    print("Accuracies : ", accuracies)

    #Accuracies :  [0.8974358974358975, 0.9102564102564102, 0.9166666666666666, 0.9665809768637532, 0.9626769626769627]

def Part3_SVM():

    accuracies=[]

    for ratio in ratios:

        clf = sklearn.svm.SVC(C=100, kernel='rbf', degree=1, gamma='auto', coef0=1, shrinking=True,
                                  probability=False,
                                  tol=1e-5, class_weight=None, decision_function_shape='ovo', random_state=None)

        print(ratio)
        indx = np.random.choice(x_train.shape[0], int(x_train.shape[0] * ratio), replace=False)
        xx = x_train[indx]
        yy = y_train[indx]
        xx_train = xx[0:int(xx.shape[0] * 0.9)]
        yy_train = yy[0:int(yy.shape[0] * 0.9)]
        xx_val = xx[int(xx.shape[0] * 0.9):]
        yy_val = yy[int(yy.shape[0] * 0.9):]
        clf.fit(xx_train, yy_train)
        accuracies.append(clf.score(xx_val, yy_val))
    print("Accuracies : ", accuracies)

    #Accuracies :  [0.8461538461538461, 0.9358974358974359, 0.9487179487179487, 0.9691516709511568, 0.9703989703989704]

def Part3_Adaboost():

    accuracies=[]

    for ratio in ratios:

        clf = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=7),
            n_estimators=50, learning_rate=1, algorithm='SAMME',
            random_state=None)

        print(ratio)
        indx = np.random.choice(x_train.shape[0], int(x_train.shape[0] * ratio), replace=False)
        xx = x_train[indx]
        yy = y_train[indx]
        xx_train = xx[0:int(xx.shape[0] * 0.9)]
        yy_train = yy[0:int(yy.shape[0] * 0.9)]
        xx_val = xx[int(xx.shape[0] * 0.9):]
        yy_val = yy[int(yy.shape[0] * 0.9):]
        clf.fit(xx_train, yy_train)
        accuracies.append(clf.score(xx_val, yy_val))
    print("Accuracies : ", accuracies)

    #Accuracies :  [0.8205128205128205, 0.9230769230769231, 0.9358974358974359, 0.9511568123393316, 0.9691119691119691]

def Part3_NN():

    accuracies=[]

    for ratio in ratios:



        print(ratio)
        indx = np.random.choice(x_train.shape[0], int(x_train.shape[0]*ratio) , replace=False)
        # xx=x_train[indx]
        # yy=y_train[indx]
        # xx_train = xx[0:int(xx.shape[0]*0.9) ]
        # yy_train = yy[0:int(yy.shape[0]*0.9) ]
        # xx_val   = xx[int(xx.shape[0]*0.9):  ]
        # yy_val   = yy[int(yy.shape[0] *0.9): ]

        #get_NN2(l1=150,l2=50,dropout1=0.2,dropout2=0.2,dropout3=0.2,batch_size=128,epochs=250,ll=10,regularization='l2')

        (model,acc)=get_NN2(x_train[indx],y_train[indx],l1=150,l2=50,dropout1=0.2,dropout2=0.2,dropout3=0.2,batch_size=128,epochs=250)


        accuracies.append(acc)
    print("Accuracies : ", accuracies)

#Accuracies =  [0.9487179487179487, 0.8717948717948718, 0.9358974358974359, 0.9665809768637532, 0.9755469755469756]
#Accuracies :  [0.8205128205128205, 0.8974358974358975, 0.9294871794871795, 0.961439588688946, 0.9626769626769627]


#############################################


feature_count=[5,10,50,100]



def Part4_Extract_Feature():



    indx = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
    xx_train = (x_train[indx])[0:int(0.9*x_train.shape[0])]
    yy_train = (y_train[indx])[0:int(0.9*x_train.shape[0])]
    xx_val = (x_train[indx])[int(0.9 * x_train.shape[0]):]
    yy_val = (y_train[indx])[int(0.9 * x_train.shape[0]):]







    print("VarinaceThreshold Method : ")
    Variance_Threshold_features = {}
    for fc in feature_count:

        tmp = sklearn.feature_selection.VarianceThreshold(threshold=np.sort(xx_train.var(axis=0))[xx_train.shape[1]-fc-1]).fit(xx_train)
        Variance_Threshold_features[fc]=tmp.get_support(True)
        np.savetxt("Feature_Selection/Variance_Threshold/"+str(fc)+".txt",Variance_Threshold_features[fc],fmt='%d')

        print("for ",fc," features : ",Variance_Threshold_features[fc])
        print()





    print("Select_Kbest_fclassif : ")
    Select_Kbest_fclassif = {}
    for fc in feature_count:

        tmp=sklearn.feature_selection.SelectKBest( sklearn.feature_selection.f_classif,k=fc  ).fit(xx_train,yy_train)
        Select_Kbest_fclassif[fc] =tmp.get_support(True)
        np.savetxt("Feature_Selection/Select_Kbest_fclassif/"+str(fc)+".txt",Select_Kbest_fclassif[fc],fmt='%d')
        print("for ", fc, " features : ", Select_Kbest_fclassif[fc])
        print()



    print("Select_Kbest_mutual_info_classif : ")
    Select_Kbest_mutual_info_classif = {}
    for fc in feature_count:

        tmp=sklearn.feature_selection.SelectKBest( sklearn.feature_selection.mutual_info_classif,k=fc  ).fit(xx_train,yy_train)
        Select_Kbest_mutual_info_classif[fc] = tmp.get_support(True)
        np.savetxt("Feature_Selection/Select_Kbest_mutual_info_classif/"+str(fc)+".txt",Select_Kbest_mutual_info_classif[fc],fmt='%d')
        print("for ", fc, " features : ", Select_Kbest_mutual_info_classif[fc])
        print()

    exit()





    print("Recursive Feature Elimination_LSVC  :  ")
    Recursive_Feature_Elimination_LSVC = {}
    for fc in feature_count:
        # clf2 = sklearn.linear_model.LogisticRegression(random_state=None, solver='newton-cg', multi_class='multinomial')
        clf2 = sklearn.svm.SVC(C=100, kernel='linear')
        tmp = sklearn.feature_selection.RFE(estimator=clf2, n_features_to_select=fc, step=1).fit(xx_train, yy_train)
        Recursive_Feature_Elimination_LSVC[fc] = tmp.get_support(True)
        np.savetxt("Feature_Selection/Recursive Feature Elimination_LSVC/" + str(fc) + ".txt",
                   Recursive_Feature_Elimination_LSVC[fc], fmt='%d')
        print("for ", fc, " features : ", Recursive_Feature_Elimination_LSVC[fc])
        print()

    exit()





    print("Select_From_Model_Tree_based  :  ")
    Select_From_Model_Tree_based = {}
    for fc in feature_count:
        clf2 = sklearn.ensemble.ExtraTreesClassifier(n_estimators=50).fit(xx_train, yy_train)
        Select_From_Model_Tree_based[fc] = sklearn.feature_selection.SelectFromModel(clf2, prefit=True, max_features=fc,
                                                                                     threshold=-np.inf).get_support(
            True)

        np.savetxt("Feature_Selection/Select_From_Model_Tree_based/" + str(fc) + ".txt",
                   Select_From_Model_Tree_based[fc],
                   fmt='%d')

        print("for ", fc, " features : ", Select_From_Model_Tree_based[fc])
        print()

    exit()



    print("Select_From_Model_L1_Based_LSVC  :  ")
    penalties = [0.01, 0.005, 0.001, 0.0005]

    for i in range(4):
        clf2 = sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=1 / penalties[i],
                                     random_state=None).fit(xx_train, yy_train)
        javab = sklearn.feature_selection.SelectFromModel(clf2, prefit=True, max_features=feature_count[i],
                                                          threshold=-np.inf).get_support(True)
        np.savetxt("Feature_Selection/Select_From_Model_L1_Based_LSVC/" + str(feature_count[i]) + ".txt", javab,
                   fmt='%d')
        print("for ", feature_count[i], " features : ", javab)
        print()

    exit()



    print("Select_From_Model_L1_Based_LR  :  ")
    penalties = [10, 10, 1, 1]

    for i in range(4):
        clf2 = sklearn.linear_model.LogisticRegression(penalty='l1', C=1 / penalties[i], random_state=None,
                                                       solver='saga', multi_class='multinomial').fit(xx_train, yy_train)
        javab = sklearn.feature_selection.SelectFromModel(clf2, prefit=True, max_features=feature_count[i],
                                                          threshold=-np.inf).get_support(True)
        np.savetxt("Feature_Selection/Select_From_Model_L1_Based_LR/" + str(feature_count[i]) + ".txt", javab, fmt='%d')
        print("for ", feature_count[i], " features : ", javab)
        print()

    exit()


def Part4_Train_Test():


    dirs=           [  'Recursive Feature Elimination_LSVC'  , 'Select_From_Model_L1_Based_LR' ,   'Select_From_Model_L1_Based_LSVC', 'Select_From_Model_Tree_based','Select_Kbest_fclassif','Select_Kbest_mutual_info_classif','Variance_Threshold']
    feature_count=  [5,10,50,100]

    clf = None

    indx = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
    xx_train = (x_train[indx])[0:int(0.9 * x_train.shape[0])]
    yy_train = (y_train[indx])[0:int(0.9 * x_train.shape[0])]
    xx_val = (x_train[indx])[int(0.9 * x_train.shape[0]):]
    yy_val = (y_train[indx])[int(0.9 * x_train.shape[0]):]

    for dir in dirs:
        print('For '+dir+" Features : ")
        accuracies=[]
        for fc in feature_count:

            features=np.loadtxt('Feature_Selection/'+dir+'/'+str(fc)+'.txt').astype(int)

            clf = sklearn.linear_model.LogisticRegression(solver='newton-cg',multi_class='multinomial')
            #clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None,min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0, max_features='auto',max_leaf_nodes=None, min_impurity_decrease=0,bootstrap=True, oob_score=False, random_state=None,warm_start=False, class_weight=None)
            #clf = sklearn.svm.SVC(C=100, kernel='rbf', degree=1, gamma='auto', coef0=1, shrinking=True,probability=False, tol=1e-5, class_weight=None, decision_function_shape='ovo',random_state=None)
            #clf = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=7),n_estimators=50, learning_rate=1, algorithm='SAMME',random_state=None)





            ###
            ###
            ###

            clf.fit(xx_train[:,features],yy_train)
            accuracies.append(clf.score(xx_val[:,features],yy_val))

        clf = sklearn.linear_model.LogisticRegression(solver='newton-cg', multi_class='multinomial')
        #clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None,min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0, max_features='auto',max_leaf_nodes=None, min_impurity_decrease=0,bootstrap=True, oob_score=False, random_state=None,warm_start=False, class_weight=None)
        #clf = sklearn.svm.SVC(C=100, kernel='rbf', degree=1, gamma='auto', coef0=1, shrinking=True,probability=False, tol=1e-5, class_weight=None, decision_function_shape='ovo',random_state=None)
        #clf = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=7),n_estimators=50, learning_rate=1, algorithm='SAMME',random_state=None)



        clf_name = 'Logistic Regression'
        #clf_name = 'Random Forest'
        #clf_name = 'SVC'
        #clf_name = 'Adaboost'

        ###
        ###
        ###

        feature_count2=[5,10,50,100,561]
        clf.fit(xx_train,yy_train)
        accuracies.append(clf.score(xx_val, yy_val))
        print('#Accuracies =',accuracies)
        print()
        print("#################################")
        ss = np.arange(5)

        plt.cla

        plt.title('Feature selection with  '+dir+' on  '+clf_name)
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.plot(ss, accuracies, 'go-')
        plt.xticks(ss, feature_count2)
        plt.show()

#############################################

def Part5_PCA():

    indx = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
    xx_train = (x_train[indx])[0:int(0.9 * x_train.shape[0])]
    yy_train = (y_train[indx])[0:int(0.9 * x_train.shape[0])]
    xx_val = (x_train[indx])[int(0.9 * x_train.shape[0]):]
    yy_val = (y_train[indx])[int(0.9 * x_train.shape[0]):]


    projector= sklearn.decomposition.PCA(n_components=10, copy=True, whiten=False,
                              svd_solver='auto', tol=0.0, iterated_power='auto',
                              random_state=None)

    xoxo=projector.fit_transform(xx_train)

    ###################


    clf_logistic_regression = sklearn.linear_model.LogisticRegression(fit_intercept=True, intercept_scaling=1,
                                                                      class_weight=None, random_state=None,
                                                                      solver='newton-cg', max_iter=100,
                                                                      multi_class='multinomial')


    clf_logistic_regression.fit(xoxo,yy_train)
    print('Logistic Regression Accuracy : ',clf_logistic_regression.score(projector.transform(xx_val),yy_val))



    ##################



    clf_random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None,
                                                                min_samples_split=2, min_samples_leaf=1,
                                                                min_weight_fraction_leaf=0, max_features='auto',
                                                                max_leaf_nodes=None, min_impurity_decrease=0,
                                                                bootstrap=True, oob_score=False, random_state=None,
                                                                warm_start=False, class_weight=None)

    clf_random_forest.fit(xoxo, yy_train)
    print('Random Forest Accuracy : ',clf_random_forest.score(projector.transform(xx_val), yy_val))

    ######################


    clf_svm = sklearn.svm.SVC(C=100, kernel='rbf', degree=1, gamma='auto', coef0=1, shrinking=True, probability=False,
                              tol=1e-5, class_weight=None, decision_function_shape='ovo', random_state=None)

    clf_svm.fit(xoxo, yy_train)
    print("SVM Accuracy : ",clf_svm.score(projector.transform(xx_val), yy_val))



    clf_adaboost = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=7),
                                                       n_estimators=50, learning_rate=1, algorithm='SAMME',
                                                       random_state=None)

    clf_adaboost.fit(xoxo, yy_train)
    print('Adaboost Accuracy : ',clf_adaboost.score(projector.transform(xx_val), yy_val))

def Part5_SVD():

    indx = np.random.choice(x_train.shape[0], x_train.shape[0], replace=False)
    xx_train = (x_train[indx])[0:int(0.9 * x_train.shape[0])]
    yy_train = (y_train[indx])[0:int(0.9 * x_train.shape[0])]
    xx_val = (x_train[indx])[int(0.9 * x_train.shape[0]):]
    yy_val = (y_train[indx])[int(0.9 * x_train.shape[0]):]


    projector=sklearn.decomposition.TruncatedSVD(n_components=10)


    xoxo=projector.fit_transform(xx_train)

    ###################


    clf_logistic_regression = sklearn.linear_model.LogisticRegression(fit_intercept=True, intercept_scaling=1,
                                                                      class_weight=None, random_state=None,
                                                                      solver='newton-cg', max_iter=100,
                                                                      multi_class='multinomial')


    clf_logistic_regression.fit(xoxo,yy_train)
    print('Logistic Regression Accuracy : ',clf_logistic_regression.score(projector.transform(xx_val),yy_val))



    ##################



    clf_random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None,
                                                                min_samples_split=2, min_samples_leaf=1,
                                                                min_weight_fraction_leaf=0, max_features='auto',
                                                                max_leaf_nodes=None, min_impurity_decrease=0,
                                                                bootstrap=True, oob_score=False, random_state=None,
                                                                warm_start=False, class_weight=None)

    clf_random_forest.fit(xoxo, yy_train)
    print('Random Forest Accuracy : ',clf_random_forest.score(projector.transform(xx_val), yy_val))

    ######################


    clf_svm = sklearn.svm.SVC(C=100, kernel='rbf', degree=1, gamma='auto', coef0=1, shrinking=True, probability=False,
                              tol=1e-5, class_weight=None, decision_function_shape='ovo', random_state=None)

    clf_svm.fit(xoxo, yy_train)
    print("SVM Accuracy : ",clf_svm.score(projector.transform(xx_val), yy_val))



    clf_adaboost = sklearn.ensemble.AdaBoostClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=7),
                                                       n_estimators=50, learning_rate=1, algorithm='SAMME',
                                                       random_state=None)

    clf_adaboost.fit(xoxo, yy_train)
    print('Adaboost Accuracy : ',clf_adaboost.score(projector.transform(xx_val), yy_val))

#######################################

def clustering_kmeans():

    k_means = sklearn.cluster.KMeans(n_clusters=12, init='random', n_init=100,
                                     max_iter=1000, tol=1e-4, precompute_distances='auto',
                                     verbose=0, random_state=None, copy_x=True,
                                     n_jobs=-1, algorithm='auto')

    k_means.fit_predict(x_train,y_train)
    ssc=k_means.score(x_train)
    pickle.dump(k_means, open('clustering/kmeans/'+str(ssc)+'.sav', 'wb'))
    print(ssc)

def load_best_kmeans():

    loaded_model = pickle.load(open('clustering/kmeans/'+'-122200.0023579341.sav', 'rb'))
    print(sklearn.metrics.adjusted_rand_score(y_train,loaded_model.predict(x_train)))


def clustering_gmm():

    gmm = sklearn.mixture.GaussianMixture(n_components=12, covariance_type='spherical', tol=1e-3,
                 reg_covar=1e-6, max_iter=300, n_init=6, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10)
    gmm.fit_predict(x_train,y_train)
    ssc=gmm.score(x_train)
    pickle.dump(gmm, open('clustering/gmm/'+str(ssc)+'.sav', 'wb'))
    print(ssc)

def load_best_gmm():

    loaded_model = pickle.load(open('clustering/gmm/'+'1345.9805174486332.sav', 'rb'))
    print(sklearn.metrics.adjusted_rand_score(y_train,loaded_model.predict(x_train)))

######################################


def pca():
    pca1= sklearn.decomposition.PCA( n_components=20, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None)

    projected = pca1.fit_transform(x_train)

    clu=sklearn.cluster.KMeans(n_clusters=12, init='random', n_init=10,
                 max_iter=500, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto')

    pr=clu.fit_predict(x_train)

    pca2 = sklearn.decomposition.PCA(n_components=2, copy=True, whiten=False,
                                     svd_solver='auto', tol=0.0, iterated_power='auto',
                                     random_state=None)

    projected2 = pca2.fit_transform(projected)

    (pr,projected2)





    plt.title('Clustering according to 20 PCA feature')
    plt.scatter(projected2[:, 0], projected2[:, 1], c=pr, s=50, cmap='viridis')
    plt.show()
    plt.cla
    plt.title('Clustering according Labels')
    plt.scatter(projected2[:, 0], projected2[:, 1], c=y_train, s=50, cmap='viridis')
    plt.show()



###############################################


Part1_2_LinearSVM_L1()

#get_NN(l1=150,l2=50,dropout1=0.2,dropout2=0.2,dropout3=0.2,batch_size=128,epochs=250,ll=10,regularization='l2')



















