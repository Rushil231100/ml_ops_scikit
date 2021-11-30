import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
import numpy as np
from joblib import dump,load
from statistics import mean,stdev
gamma_array =  [1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]
def create_split(data_x,data_y, train_part=70,test_part = 20 ,val_part=10):
    
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=((test_part+val_part)/ (train_part+test_part + val_part)), shuffle=True)
    #print("before ",len(X_train),len(X_test))
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=((val_part)/(test_part+val_part)), shuffle=True)
    
    return X_train, X_test,  X_val, y_train, y_test,y_val 

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test,  X_val, y_train, y_test,y_val  = create_split(data,digits.target,80,10 ,10)
maxi =0
argmax_gamma_model ={}
best_model = 0
AFR_array =[]
line_chart_y =[]
for i in range(10,110,10):
    ind = int(len(X_train)*i/100)
    X_train_i = X_train[:ind]
    y_train_i = y_train[:ind]
    print(f"\nFOR {i}% ... ",end='')
    for g in gamma_array:
        clf = svm.SVC(gamma=g)
        clf.fit(X_train_i, y_train_i)
        predicted = clf.predict(X_val)
        f1 = metrics.f1_score(y_val, predicted,average = 'macro')
        
        if(maxi < f1):
            maxi = f1
            argmax_gamma_model = g
            best_model = clf
    # maxi*=100
    predicted = clf.predict(X_test)
    F1_test= metrics.f1_score(y_test, predicted,average = 'macro')
    line_chart_y.append(F1_test)
    maxi = round(maxi,2)
    F1_test = round(F1_test,2)
    print(f"Best model is at gamma = {argmax_gamma_model},\n\tValidation macro-F1 = {maxi}%; Test macro-F1 = {F1_test}")
    cnf_matrix = confusion_matrix(y_test, predicted)
    # print(cnf_matrix)
    #[[1 1 3]
    # [3 2 2]
    # [1 3 1]]

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    AFR = (FPR+FNR)/2
    # print("Average false rate = ", np.round(AFR,2))
    print(" \t Mean AFR =", round(sum(AFR)/10,4))
    AFR_array.append(np.round(AFR,2))
# fig, ax = plt.subplots()
    #ax.hist(a, bins = [0, 25, 50, 75, 100])
plt.plot(range(10,110,10), line_chart_y) #[7,7.5, 8, 8.5, 9,9.5,10,10.5,11,11.5]
plt.xlabel('%% of training set used')
plt.ylabel('macro-F1 on test set')
plt.title('macro-F1 of test set VS %%training set used ')
plt.savefig('macro-F1_of_test_set_VS_%%training_set_used.png')
plt.show()

for i in range(9):
    plt.bar(range(10,110,10), AFR_array[i],color = 'maroon',width = 4) #[7,7.5, 8, 8.5, 9,9.5,10,10.5,11,11.5]
    plt.bar(range(10,110,10), AFR_array[i+1],color = 'blue',alpha = 0.6,width = 4)
    plt.xlabel('Class Label')
    plt.ylabel('Average False Rate')
    plt.title('Average False Rate VS each class of MNIST ')
    plt.savefig(str(10*(i+2))+"%_vs"+str(10*(i+1))+'%.png')
    plt.show()
        