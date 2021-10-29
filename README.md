Comparing SVM  with Decision tree.

Testing on following gamma values of SVM = [1,0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]

following max_depth in decision tree =  [2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80]

This test has been carried 5 times. In each iteration , the train-val-test ratio is 70-20-10. The shuffling is set to True.

Table is for classification accuracy .

Sr,No.		SVM_acc	gamma		Dtree	depth
------------------------------------------------------------------------------------
0		      0.98	0.003		0.82	14
1		      0.99	0.001		0.86	12
2		      0.99	0.001		0.83	40
3		      0.98	0.001		0.8	  12
4		      0.99	0.001		0.82	8
------------------------------------------------------------------------------------
mean,std	 0.986 +- 0.005 	 0.826 +- 0.022

Observation : 

SVM is performing better than Decision Tree in all the 5 cases. The mean accuracy is 16% higher, and relatively 20% hgher than the Dtree accuracy.
Also the Std is less for SVM, that shown it is consistent accross gamma values.



  

