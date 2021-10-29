
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from joblib import dump,load
from statistics import mean,stdev

test_to_train_ratio = [0.2]
image_resolution = [8]
gamma_array =  [1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]
depth_array = [2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80]
argmax_gamma_model = {}

def create_split(data_x,data_y, train_part=70,test_part = 20 ,val_part=10):
    
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=((test_part+val_part)/ (train_part+test_part + val_part)), shuffle=True)
    #print("before ",len(X_train),len(X_test))
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=((val_part)/(test_part+val_part)), shuffle=True)
    
    return X_train, X_test,  X_val, y_train, y_test,y_val 
    
def get_accuracy(imgs,expeted_model_file,train_part=70,test_part = 20 ,val_part=10):
    data = imgs.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    

    # Split data into 50% train and 50% test subsets
    #X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=test_to_train_ratio, shuffle=False)
    # print(X_train.shape)
    # Learn the digits on the train subset
    #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_to_test_ratio, shuffle=False)
    X_train, X_test,  X_val, y_train, y_test,y_val  = create_split(data,digits.target,train_part,test_part ,val_part)
    maxi=0
    global argmax_gamma_model
    for g in gamma_array:
        clf = svm.SVC(gamma=g)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_val)
        accuracy = metrics.accuracy_score(y_val, predicted)
        if(maxi < accuracy):
            maxi = accuracy
            argmax_gamma_model["model_name"] = expeted_model_file#"models/best_accu_{}_gamma_{}_model.joblib".format(accuracy,g)
            argmax_gamma_model["val_accu"] = accuracy
            argmax_gamma_model["gamma"] = g
            
            #print(argmax_gamma)
    # print(argmax_gamma)
    # clf = svm.SVC(gamma=argmax_gamma)
    # clf.fit(X_train, y_train)
    # Predict the value of the digit on the test subset
    dump(clf,argmax_gamma_model["model_name"])
    clf = load(argmax_gamma_model["model_name"])
    predicted = clf.predict(X_test)


    
    # print(round(metrics.accuracy_score(y_test, predicted),4))
    train_metrics ={}
    train_metrics['acc'] = round(100*metrics.accuracy_score(y_test, predicted),2)
    train_metrics['f1'] = round(metrics.f1_score(y_test, predicted, average='macro'),2) 
    
    return train_metrics #, round(sklearn.metrics.f1_score(y_test, predicted, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'),2)
    
    
    
def run_classification_experiment(train,val,expeted_model_file):
    #X_train, X_test,  X_val, y_train, y_test,y_val  = create_split(data,digits.target,train_part,test_part ,val_part)
    
    X_train, X_test,  X_val, y_train, y_test,y_val = train.images,val.images,val.images,train.target,val.target,val.target
    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_train), -1))
    X_val = X_val.reshape((len(X_train), -1))
    maxi=0
    global argmax_gamma_model
    for g in gamma_array:
        clf = svm.SVC(gamma=g)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_val)
        accuracy = metrics.accuracy_score(y_val, predicted)
        if(maxi < accuracy):
            maxi = accuracy
            argmax_gamma_model["model_name"] = expeted_model_file#"models/best_accu_{}_gamma_{}_model.joblib".format(accuracy,g)
            argmax_gamma_model["val_accu"] = accuracy
            argmax_gamma_model["gamma"] = g
    dump(clf,argmax_gamma_model["model_name"])
    clf = load(argmax_gamma_model["model_name"])
    predicted = clf.predict(X_test)
    train_metrics ={}
    train_metrics['acc'] = round(100*metrics.accuracy_score(y_test, predicted),2)
    train_metrics['f1'] = round(metrics.f1_score(y_test, predicted, average='macro'),2) 
    print(train_metrics)
    return train_metrics


def model_accuracy(model_name,X_train, X_test,  X_val, y_train, y_test,y_val,hyperparameter=1):
    #X_train, X_test,  X_val, y_train, y_test,y_val  = create_split(data,digits.target,train_part,test_part ,val_part)
    
    # X_train, X_test,  X_val, y_train, y_test,y_val = create_split(data_x,data_y)# train.images,test.images,val.images,train.target,test.target,val.target
    clf = 0
    if(model_name=="SVM"):
        clf = svm.SVC(gamma=hyperparameter)
    if(model_name=="Dtree"):
        clf =  tree.DecisionTreeClassifier(max_depth = hyperparameter)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_val)
    accuracy_val = metrics.accuracy_score(y_val, predicted)
    predicted = clf.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, predicted)
    return accuracy_val,accuracy_test,hyperparameter

def compare_svm_dtree(data_x,data_y):
    X_train, X_test,  X_val, y_train, y_test,y_val = create_split(data_x,data_y)
    X_train = X_train.reshape((len(X_train), -1))
    X_test = X_test.reshape((len(X_test), -1))
    X_val = X_val.reshape((len(X_val), -1))
    maxi=0
    SVM_accuracy=0
    SVM_best_gamma = 0
    Dtree_accuracy=0
    Dtree_best_depth = 0
    for g in gamma_array:
        accuracy_val,accuracy_test,hyperparameter = model_accuracy("SVM",X_train, X_test,  X_val, y_train, y_test,y_val,hyperparameter=g)
        if(accuracy_val>maxi):
            maxi = accuracy_val
            SVM_accuracy = accuracy_test
            SVM_best_gamma = hyperparameter
    maxi =0
    for depth in depth_array:
        accuracy_val,accuracy_test,hyperparameter = model_accuracy("Dtree",X_train, X_test,  X_val, y_train, y_test,y_val,hyperparameter=depth)
        if(accuracy_val>maxi):
            maxi = accuracy_val
            Dtree_accuracy = accuracy_test
            Dtree_best_depth = hyperparameter
    return round(SVM_accuracy,2),SVM_best_gamma,round(Dtree_accuracy,2),Dtree_best_depth

def print_table_comparision(freq = 5):
    svm_array=[]
    dtree_array=[]
    digits = datasets.load_digits()
    print("Sr,No.\t","SVM_acc","gamma\t","Dtree","depth",sep="\t")
    print("------------------------------------------------------------------------------------")
    for i in range(freq):
        SVM_accuracy,SVM_best_gamma,Dtree_accuracy,Dtree_best_depth = compare_svm_dtree(digits.images,digits.target)
        svm_array.append(SVM_accuracy)
        dtree_array.append(Dtree_accuracy)
        print(i,"",SVM_accuracy,SVM_best_gamma,"",Dtree_accuracy,Dtree_best_depth,sep='\t')
    print("------------------------------------------------------------------------------------")
    print("mean,std\t",mean(svm_array),"+-",round(stdev(svm_array),3),"\t",mean(dtree_array),"+-",round(stdev(dtree_array),3))

print_table_comparision(5)
    # global argmax_gamma_model
    # for g in gamma_array:
    #     clf = svm.SVC(gamma=g)
    #     clf.fit(X_train, y_train)
    #     predicted = clf.predict(X_val)
    #     accuracy = metrics.accuracy_score(y_val, predicted)
    #     if(maxi < accuracy):
    #         maxi = accuracy
    #         argmax_gamma_model["model_name"] = expeted_model_file#"models/best_accu_{}_gamma_{}_model.joblib".format(accuracy,g)
    #         argmax_gamma_model["val_accu"] = accuracy
    #         argmax_gamma_model["gamma"] = g
    # dump(clf,argmax_gamma_model["model_name"])
    # clf = load(argmax_gamma_model["model_name"])
    # predicted = clf.predict(X_test)
    # train_metrics ={}
    # train_metrics['acc'] = round(100*metrics.accuracy_score(y_test, predicted),2)
    # train_metrics['f1'] = round(metrics.f1_score(y_test, predicted, average='macro'),2) 
    # print(train_metrics)
    # return train_metrics
#plt.show()

# them using :func:`matplotlib.pyplot.imread`.



# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)



# flatten the images
#test_to_train_ratio = [0.1,0.2,0.3]
#image_resolution = [64,32,8]

# n_samples = len(digits.images)

# print(imgs.shape)
# print(n_samples,digits.images.shape,get_accuracy(test_to_train_ratio,image_resolution))

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes,imgs, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)
# print(test_to_train_ratio,image_resolution,get_accuracy(test_to_train_ratio,imgs))
#print("Gamma_Value-->\tAccuracy ")
##print("================================================")
# print(argmax_gamma)
#best_accu = get_accuracy(digits.images,70,20,10)
# print(argmax_gamma)
#print(argmax_gamma_model["gamma"] ,"   -->   ",best_accu)
# for index,gamma_val in enumerate(gamma_array):
#     for i in image_resolution :
#         imgs = np.empty((n_samples,i,i))
#         for k in range(n_samples):
#             imgs[k] = resize(digits.images[k], (i,i),anti_aliasing=True)
#         for j in test_to_train_ratio:
#         #print(str(i)+"x"+str(i)+"    -->\t",str(int(100-(100*j)))+":"+str(int(100*j))+"    -->\t",get_accuracy(j,imgs,gamma_val),"%",sep='')
#             print(gamma_val,"\t",get_accuracy(j,0.1,imgs))
#             print()
# plt.show()
