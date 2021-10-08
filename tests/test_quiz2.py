from ml_ops_scikit import plot_digits_classification
import numpy as np
print("imported")

def test_create_split_1():
    n=100
    train_part = 70
    test_part = 20
    val_part = 10
    data_x = np.empty(n)
    data_y = np.empty(n)
    X_train, X_test,  X_val, y_train, y_test,y_val  = plot_digits_classification.create_split(data_x,data_y,train_part,test_part ,val_part)
    
    #print(len(X_train),len(X_test),len(X_val))
    assert len(X_train) == 70 
    assert len(X_test) == 20
    assert len(X_val) == 10
    assert len(X_train) +len(X_test)+len(X_val)==100
    
def test_create_split_2():
    n=9
    train_part = 70
    test_part = 20
    val_part = 10
    data_x = np.empty(n)
    data_y = np.empty(n)
    X_train, X_test,  X_val, y_train, y_test,y_val  = plot_digits_classification.create_split(data_x,data_y,train_part,test_part ,val_part)
    
    #print(len(X_train),len(X_test),len(X_val))
    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(X_val) == 1
    assert len(X_train) +len(X_test)+len(X_val)==9
    

    
