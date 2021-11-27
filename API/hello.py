from flask import Flask
from flask import request
from joblib import dump,load
import numpy as np
app = Flask(__name__)
model_svm = load('/home/rushil/Desktop/ml_ops_scikit/ml_ops_scikit/models/best_accu_0.9559585492227979_gamme_0.001_model.joblib')
model_dtree = load('/home/rushil/Desktop/ml_ops_scikit/ml_ops_scikit/models/Dtree12.joblib')
class_name_index = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9}
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/rushil")
def hello_rushil():
    return "<p>Hi Rushil !!</p>"

@app.route('/svm_predict',methods=['POST'])
def predict_svm():
    input_json = request.json
    image = input_json['image']
    # print(image)
    
    image = np.array(image).reshape(1,-1)
    # print(image)
    prediction = model_svm.predict(image)
    output = class_name_index[str(prediction[0])]
    output_str = "\nTesting on SVM : Prediction Class = "+ str(output)
    print(output_str)
    # return None
    html_output = "<p>"+output_str+"<\p>"
    return html_output

@app.route('/decision_tree_predict',methods=['POST'])
def predict_dtree():
    input_json = request.json
    image = input_json['image']
    # print(image)
    
    image = np.array(image).reshape(1,-1)
    # print(image)
    prediction = model_dtree.predict(image)
    output = class_name_index[str(prediction[0])]
    output_str = "\nTesting on Dtree : Prediction Class = "+ str(output)
    print(output_str)
    # return None
    html_output = "<p>"+output_str+"<\p>"
    return html_output

