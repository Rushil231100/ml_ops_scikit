Comparing SVM  with Decision tree.

Testing on following gamma values of SVM = [1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]

following max_depth in decision tree =  [2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80]

This test has been carried 5 times. In each iteration , the train-val-test ratio is 70-20-10. The shuffling is set to True.

Table is for classification accuracy .

![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/comparing_models/images/Screenshot%20from%202021-10-29%2011-52-01.png)


Observation : 

SVM is performing better than Decision Tree in all the 5 cases. The mean accuracy is 16% higher, and relatively 20% hgher than the Dtree accuracy.
Also the Std is less for SVM, that shown it is consistent accross gamma values.



  

