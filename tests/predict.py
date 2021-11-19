from joblib import dump,load
import numpy as np

model = load('/home/rushil/Desktop/ml_ops_scikit/ml_ops_scikit/models/best_accu_0.9559585492227979_gamme_0.001_model.joblib')
class_name_index = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9}
model_dtree = load('/home/rushil/Desktop/ml_ops_scikit/ml_ops_scikit/models/Dtree12.joblib')
def predict_image_on_SVM(image):
    image = np.array(image).reshape(1,-1)
    # print(image)
    prediction = model.predict(image)
    output = class_name_index[str(prediction[0])]
    output_str = "\nPrediction Class = "+ str(output)
    #print(output_str)
    # return None
    #html_output = "<p>"+output_str+"<\p>"
    return output#integer

def predict_image_on_dtree(image):
    image = np.array(image).reshape(1,-1)
    # print(image)
    prediction = model_dtree.predict(image)
    output = class_name_index[str(prediction[0])]
    output_str = "\nPrediction Class = "+ str(output)
    #print(output_str)
    # return None
    #html_output = "<p>"+output_str+"<\p>"
    return output#integer