import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from joblib import dump,load
from statistics import mean,stdev

def create_split(data_x,data_y, train_part=70,test_part = 20 ,val_part=10):
    
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=((test_part+val_part)/ (train_part+test_part + val_part)), shuffle=True)
    #print("before ",len(X_train),len(X_test))
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=((val_part)/(test_part+val_part)), shuffle=True)
    
    return X_train, X_test,  X_val, y_train, y_test,y_val 

depth_array = [2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80]
hyperparameter_array = [
    [10,20,5],
    [100,20,5],
    [10,20,500],
    [500,2,500],
    [100,2,5]
]
digits = datasets.load_digits()
# X_train, X_test,  X_val, y_train, y_test,y_val  = create_split(digits.images,digits.target, train_part=70,test_part = 15 ,val_part=15)
# print(X_train.shape,y_train.shape)
# clf =  tree.DecisionTreeClassifier(max_leaf_nodes= alpha, min_samples_split = beta,  max_depth = gamma)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_train)
split_whole = [create_split(digits.images,digits.target, train_part=70,test_part = 15 ,val_part=15),create_split(digits.images,digits.target, train_part=70,test_part = 15 ,val_part=15),create_split(digits.images,digits.target, train_part=70,test_part = 15 ,val_part=15)]
#print("a b c ",beta,gamma," \t",accuracy_train[0],accuracy_val[0],accuracy_test[0]," \t",accuracy_train[1],accuracy_val[1],accuracy_test[1]," \t",accuracy_train[2],accuracy_val[2],accuracy_test[2]," \t",np.around(accuracy_train.mean(),2),np.around(accuracy_val.mean(),2),np.around(accuracy_test.mean(),2))
print("\na   b   c \t  Train  Dev  Test  \t  Train  Dev  Test \t  Train  Dev  Test \t  Train  Dev  Test \t")
print("==========================================================================================================================")
for alpha,beta,gamma in hyperparameter_array:
    accuracy_train = []
    accuracy_val =[]
    accuracy_test =[]
    for i in range(3):
        X_train, X_test,  X_val, y_train, y_test,y_val  = split_whole[0]
        X_train = X_train.reshape((len(X_train), -1))
        X_test = X_test.reshape((len(X_test), -1))
        X_val = X_val.reshape((len(X_val), -1))
        # print(X_train.shape,y_train.shape)
        clf =  tree.DecisionTreeClassifier(max_leaf_nodes= alpha, min_samples_split = beta,  max_depth = gamma)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_train)
        accuracy_train.append(metrics.accuracy_score(y_train, predicted))
        predicted = clf.predict(X_val)
        accuracy_val.append(metrics.accuracy_score(y_val, predicted))
        predicted = clf.predict(X_test)
        accuracy_test.append(metrics.accuracy_score(y_test, predicted))
    
    accuracy_train = np.around(np.array(accuracy_train),2)
    accuracy_val =np.around(np.array(accuracy_val),2)
    accuracy_test =np.around(np.array(accuracy_test),2)
    
    print(alpha,beta,gamma," \t",accuracy_train[0],accuracy_val[0],accuracy_test[0]," \t",accuracy_train[1],accuracy_val[1],accuracy_test[1]," \t",accuracy_train[2],accuracy_val[2],accuracy_test[2]," \t",np.around(accuracy_train.mean(),2),np.around(accuracy_val.mean(),2),np.around(accuracy_test.mean(),2),sep='  ')
    print("\n\n\n")