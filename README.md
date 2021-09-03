# Quiz 1 ml_ops_scikit 
This is a readme file for feature/quiz breanch

#RESULTS Obtained
--Rushil Sanghavi (B18CSE066)

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9. 
It shows the variation of accuracy with respect to change in image size and train test ratio. 

Image.size -->	Train-Test -->	Accuracy \n
================================================\n
64x64    -->	90:10    -->	12.78% \n
64x64    -->	80:20    -->	13.89% \n
64x64    -->	70:30    -->	11.3%  \n

32x32    -->	90:10    -->	78.33% \n
32x32    -->	80:20    -->	77.5%\n
32x32    -->	70:30    -->	75.74%\n

8x8    -->		90:10    -->	96.11%\n
8x8    -->		80:20    -->	95.83%\n
8x8    -->		70:30    -->	97.04%\n


#OBSERVATIONS

1. Increasing the size of the image , decreases the accuracy drastically. As we see from 8x8 to 64x64 the accuracy is dropping upto ~80%\n
2. For the dataset of same image resolutionn, if we change the train-test split ratio, accuracy does not depend on it. From the results above, there is no uniformity in selecting better train test ratio.
  

