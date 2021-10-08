"""
================================
            QUIZ 2
================================
--Rushil Sanghavi (B18CSE066)

This example shows how testing of a targetted function works.

"""
#create a folder name models, add it to gitignore, save all models in it, load the best one olny, and refactor code with different functions
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
from skimage.transform import resize
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump,load

test_to_train_ratio = [0.2]
image_resolution = [8]
gamma_array = [1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]
argmax_gamma_model = {}

def create_split(data_x,data_y, train_part=70,test_part = 20 ,val_part=10):
    
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=((test_part+val_part)/ (train_part+test_part + val_part)), shuffle=False)
    #print("before ",len(X_train),len(X_test))
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=((val_part)/(test_part+val_part)), shuffle=False)
    
    return X_train, X_test,  X_val, y_train, y_test,y_val 
    
def get_accuracy(imgs,train_part=70,test_part = 20 ,val_part=10):
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
            argmax_gamma_model["model_name"] = "models/best_accu_{}_gamma_{}_model.joblib".format(accuracy,g)
            argmax_gamma_model["val_accu"] = accuracy
            argmax_gamma_model["gamma"] = g
            dump(clf,argmax_gamma_model["model_name"])
            #print(argmax_gamma)
    # print(argmax_gamma)
    # clf = svm.SVC(gamma=argmax_gamma)
    # clf.fit(X_train, y_train)
    # Predict the value of the digit on the test subset
    clf = load(argmax_gamma_model["model_name"])
    predicted = clf.predict(X_test)

    ###############################################################################
    # Below we visualize the first 4 test samples and show their predicted
    # digit value in the title.

    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test, predicted):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     ax.set_title(f'Prediction: {prediction}')

    ###############################################################################
    # :func:`~sklearn.metrics.classification_report` builds a text report showing
    # the main classification metrics.

    # print(f"Classification report for classifier {clf}:\n"
    #       f"{metrics.classification_report(y_test, predicted)}\n")
    # print(round(metrics.accuracy_score(y_test, predicted),4))
    return round(100*metrics.accuracy_score(y_test, predicted),2) #, round(sklearn.metrics.f1_score(y_test, predicted, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn'),2)
###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

# disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()
###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)


###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
#test_to_train_ratio = [0.1,0.2,0.3]
#image_resolution = [64,32,8]

n_samples = len(digits.images)

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
