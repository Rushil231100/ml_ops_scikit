import math
from ml_ops_scikit import plot_digits_classification
import numpy as np
from sklearn import datasets, svm, metrics
import os
def test_square():
    assert 7*7 == 49
    
def test_model_writing():
#    #creating data
    expeted_model_file = "model_weights_testing_lambda_1.joblib"
    digits = datasets.load_digits()
    plot_digits_classification.get_accuracy(digits.images, expeted_model_file)
    assert os.path.isfile(expeted_model_file)

def void_test_small_data_overfit_checking():
    digits = datasets.load_digits()
    print(len(digits.images))
    digits.images = digits.images[:100] 
    digits.target = digits.target[:100]
    expeted_model_file = "model_weights_testing_lambda_1.joblib"
    train_metrics = plot_digits_classification.run_classification_experiment(train=digits, val=digits,expeted_model_file= expeted_model_file)

    #1. create a small amount of data / (digits / subsampling)
	
    #2. train_metrics = run_classification_experiment(train=train, valid=train)
    threshold_acc = 75
    threshold_f1 = 0.5
    assert train_metrics['acc']  > threshold_acc

    assert train_metrics['f1'] > threshold_f1

    
