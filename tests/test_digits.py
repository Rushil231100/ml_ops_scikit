import numpy as np
from sklearn import datasets, svm, metrics
import os
from tests import predict #this is locally mported created by me to predict iages


digits = datasets.load_digits()
#print(len(digits.images))

#BONUS
min_acc_req = 70 #minimum 70% per class
def test_class_wise_accuracy_svm():
    acc_digit=[]
    for i in range (10):
        count = 0
        images = digits.images[digits.target == i] #subsampling only images with label i
        expected_output = digits.target[digits.target == i]
        for ind in range(len(images)): 
            predicted_output = predict.predict_image_on_SVM(images[ind]) #predicting only one image out of this subsample
            count += (predicted_output==expected_output[ind])
        acc_digit.append(100*count/len(images))
    
    assert acc_digit[0] > min_acc_req
    assert acc_digit[1] > min_acc_req
    assert acc_digit[2] > min_acc_req
    assert acc_digit[3] > min_acc_req
    assert acc_digit[4] > min_acc_req
    assert acc_digit[5] > min_acc_req
    assert acc_digit[6] > min_acc_req
    assert acc_digit[7] > min_acc_req
    assert acc_digit[8] > min_acc_req
    assert acc_digit[9] > min_acc_req
    print("\n[BONUS] \tSVM has passed minimum classwise accuracy assertion at 70% per class")

def test_class_wise_accuracy_dtree():
    acc_digit=[]
    for i in range (10):
        count = 0
        images = digits.images[digits.target == i] #subsampling only images with label i
        expected_output = digits.target[digits.target == i]
        for ind in range(len(images)): 
            predicted_output = predict.predict_image_on_dtree(images[ind]) #predicting only one image out of this subsample
            count += (predicted_output==expected_output[ind])
        acc_digit.append(100*count/len(images))
    
    assert acc_digit[0] > min_acc_req
    assert acc_digit[1] > min_acc_req
    assert acc_digit[2] > min_acc_req
    assert acc_digit[3] > min_acc_req
    assert acc_digit[4] > min_acc_req
    assert acc_digit[5] > min_acc_req
    assert acc_digit[6] > min_acc_req
    assert acc_digit[7] > min_acc_req
    assert acc_digit[8] > min_acc_req
    assert acc_digit[9] > min_acc_req
    print("[BONUS] \t DTREE has passed minimum classwise accuracy assertion at 70% per class")

def test_digit_correct_svm_0():
    images_0 = digits.images[digits.target == 0] #subsampling only images with label 0 
    predicted_output = predict.predict_image_on_SVM(images_0[0]) #predicting only one image out of this subsample
    expected_output = digits.target[digits.target == 0][0] #extractin the corresponding  truth label
    assert  predicted_output==expected_output
    print("SVM test for digit 0 passed")

def test_digit_correct_svm_1():
    images_1 = digits.images[digits.target == 1] 
    predicted_output = predict.predict_image_on_SVM(images_1[0])
    expected_output = digits.target[digits.target == 1][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 1 passed")

def test_digit_correct_svm_2():
    images_2 = digits.images[digits.target == 2] 
    predicted_output = predict.predict_image_on_SVM(images_2[0])
    expected_output = digits.target[digits.target == 2][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 2 passed")

def test_digit_correct_svm_3():
    images_3 = digits.images[digits.target == 3] 
    predicted_output = predict.predict_image_on_SVM(images_3[0])
    expected_output = digits.target[digits.target == 3][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 3 passed")

def test_digit_correct_svm_4():
    images_4 = digits.images[digits.target == 4] 
    predicted_output = predict.predict_image_on_SVM(images_4[0])
    expected_output = digits.target[digits.target == 4][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 4 passed")

def test_digit_correct_svm_5():
    images_5 = digits.images[digits.target == 5] 
    predicted_output = predict.predict_image_on_SVM(images_5[1])
    expected_output = digits.target[digits.target == 5][1]
    assert  predicted_output==expected_output
    print("SVM test for digit 5 passed")
def test_digit_correct_svm_6():
    images_6 = digits.images[digits.target == 6] 
    predicted_output = predict.predict_image_on_SVM(images_6[0])
    expected_output = digits.target[digits.target == 6][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 6 passed")

def test_digit_correct_svm_7():
    images_7 = digits.images[digits.target == 7] 
    predicted_output = predict.predict_image_on_SVM(images_7[0])
    expected_output = digits.target[digits.target == 7][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 7 passed")
def test_digit_correct_svm_8():
    images_8 = digits.images[digits.target == 8] 
    predicted_output = predict.predict_image_on_SVM(images_8[0])
    expected_output = digits.target[digits.target == 8][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 8 passed")

def test_digit_correct_svm_9():
    images_9 = digits.images[digits.target == 9] 
    predicted_output = predict.predict_image_on_SVM(images_9[0])
    expected_output = digits.target[digits.target == 9][0]
    assert  predicted_output==expected_output
    print("SVM test for digit 9 passed")

def test_digit_correct_dtree_0():
    images_0 = digits.images[digits.target == 0] #subsampling only images with label 0 
    predicted_output = predict.predict_image_on_dtree(images_0[0]) #predicting only one image out of this subsample
    expected_output = digits.target[digits.target == 0][0] #extractin the corresponding  truth label
    assert  predicted_output==expected_output
    print("Dtree test for digit 0 passed")

def test_digit_correct_dtree_1():
    images_1 = digits.images[digits.target == 1] 
    predicted_output = predict.predict_image_on_dtree(images_1[0])
    expected_output = digits.target[digits.target == 1][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 1 passed")

    
def test_digit_correct_dtree_2():
    images_2 = digits.images[digits.target == 2] 
    predicted_output = predict.predict_image_on_dtree(images_2[0])
    expected_output = digits.target[digits.target == 2][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 2 passed")


def test_digit_correct_dtree_3():
    images_3 = digits.images[digits.target == 3] 
    predicted_output = predict.predict_image_on_dtree(images_3[0])
    expected_output = digits.target[digits.target == 3][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 3 passed")


def test_digit_correct_dtree_4():
    images_4 = digits.images[digits.target == 4] 
    predicted_output = predict.predict_image_on_dtree(images_4[0])
    expected_output = digits.target[digits.target == 4][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 4 passed")


def test_digit_correct_dtree_5():
    images_5 = digits.images[digits.target == 5] 
    predicted_output = predict.predict_image_on_dtree(images_5[1])
    expected_output = digits.target[digits.target == 5][1]
    assert  predicted_output==expected_output
    print("Dtree test for digit 5 passed")

def test_digit_correct_dtree_6():
    images_6 = digits.images[digits.target == 6] 
    predicted_output = predict.predict_image_on_dtree(images_6[0])
    expected_output = digits.target[digits.target == 6][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 6 passed")


def test_digit_correct_dtree_7():
    images_7 = digits.images[digits.target == 7] 
    predicted_output = predict.predict_image_on_dtree(images_7[0])
    expected_output = digits.target[digits.target == 7][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 7 passed")

def test_digit_correct_dtree_8():
    images_8 = digits.images[digits.target == 8] 
    predicted_output = predict.predict_image_on_dtree(images_8[0])
    expected_output = digits.target[digits.target == 8][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 8 passed")


def test_digit_correct_dtree_9():
    images_9 = digits.images[digits.target == 9] 
    predicted_output = predict.predict_image_on_dtree(images_9[0])
    expected_output = digits.target[digits.target == 9][0]
    assert  predicted_output==expected_output
    print("Dtree test for digit 9 passed")


