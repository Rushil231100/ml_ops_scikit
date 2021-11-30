Assignment 11
This is a readme file for feature/quiz3 branch

# RESULTS Obtained
--Rushil Sanghavi (B18CSE066)  
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/macro-F1_of_test_set_VS_%25%25training_set_used.png)

Observation :   
From the graph above, it can be inferred that the macro-F1 score increases as we train our model with more training images. Ideally, macro-F1 should be 1 , which we can see that the graph is approaching towards 1. At each step the model is performing better or atleast as good as ts previos one, which was trained on less training set.   
  
Evaluation Metric for comparision : Average False Rate (AFR)  
>>> AFR =  ( False positive rate + False Negetive Rate ) / 2
AFR can be obtained from confusion matrix.
Since there are 10 classes, AFR will be class wise. i.e it will be a vector of size (10,).  To compare with a single metric, I took mean of AFR along the classes.  
For the model trained with i% of samples, AFR is a good metric to compare.   

REASON : If the model is trained on less samples, there are high chances of BIAS in it. Along with bias, it also have high chances of FN and FP on a test set.  
The AFR metric captures the notion of Bias and False rate together.
>>> Note : Lesser the AFR is better the model.
  
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/Screenshot%20from%202021-12-01%2000-20-52.png)
Comparing 20% vs 10 %  
From the figure given below, maoon is the AFR of 10% and blue colour is AFR for 20 % . 
We can see that blue bar chart is lower or equal to maroon in most the cases. That means AFR is less than it's 10% model.
  
Similarly, Looking at all the graphs given below, one thing is for sure , 
As the training samples increases, the AFR of model is lesser or equal to its previous one. It is gurateed that in this case mean of AFR is always lesser.  
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/20%25_vs10%25.png)  
  
Comparing 30% vs 20 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/30%25_vs20%25.png)
  
Comparing 40% vs 30 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/40%25_vs30%25.png)
  
Comparing 50% vs 40 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/50%25_vs40%25.png)
  
Comparing 60% vs 50 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/60%25_vs50%25.png)
  
Comparing 70% vs 60 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/70%25_vs60%25.png)
  
Comparing 80% vs 70 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/80%25_vs70%25.png)
  
Comparing 90% vs 80 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/90%25_vs80%25.png)
  
Comparing 100% vs 90 %
![alt text](https://github.com/Rushil231100/ml_ops_scikit/blob/features/assignment_11/100%25_vs90%25.png)


  

