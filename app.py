from flask import Flask, render_template, request
import requests
import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import sys




app = Flask(__name__)
model = pickle.load(open('svm.sav', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

scaler = StandardScaler()
to_get_dummy=['sex','cp','fbs','restecg','exang','slope','ca','thal']
to_scale=['age','trestbps','chol','thalach','oldpeak']

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data={}
        data['age'] = int(request.form['age'])
        if int(request.form['sex']) == 1:
            data['sex_0'] = 1
            data['sex_1'] = 0
        else:
            data['sex_0'] = 0
            data['sex_1'] = 1
    
        if int(request.form['cp']) == 0:
            data['cp_0']=1
            data['cp_1']=0
            data['cp_2']=0
            data['cp_3']=0
        elif int(request.form['cp']) == 1:
            data['cp_0']=0
            data['cp_1']=1
            data['cp_2']=0
            data['cp_3']=0
        elif int(request.form['cp']) == 2:
            data['cp_0']=0
            data['cp_1']=0
            data['cp_2']=1
            data['cp_3']=0
        else:
            data['cp_0']=0
            data['cp_1']=0
            data['cp_2']=0
            data['cp_3']=1
            
        data['trestbps'] = int(request.form['trestbps'])
        data['chol'] = int(request.form['chol'])
        
        if int(request.form['fbs']) == 0:
            data['fbs_0']=1
            data['fbs_1']=0
        else:
            data['fbs_0']=0
            data['fbs_1']=1
        
        if int(request.form['restecg']) == 0:
            data['restecg_0'] = 1
            data['restecg_1'] = 0
            data['restecg_2'] = 0
        elif int(request.form['restecg']) == 1:
            data['restecg_0'] = 0
            data['restecg_1'] = 1
            data['restecg_2'] = 0
        else:
            data['restecg_0'] = 0
            data['restecg_1'] = 0
            data['restecg_2'] = 1
            
        
        data['thalach'] = int(request.form['thalach'])
        
        
        if  int(request.form['exang']) == 0:
            data['exang_0'] = 1 
            data['exang_1'] = 0
        else:
            data['exang_0'] = 0
            data['exang_1'] = 1
        
        data['oldpeak'] = float(request.form['oldpeak'])
        
        if int(request.form['slope']) == 0:
            data['slope_0']=1
            data['slope_1']=0
            data['slope_2']=0
        elif int(request.form['slope']) == 1:
            data['slope_0']=0
            data['slope_1']=1
            data['slope_2']=0
        else:
            data['slope_0']=0
            data['slope_1']=0
            data['slope_2']=1
        
        if int(request.form['ca'])==0:
            data['ca_0'] = 1
            data['ca_1'] = 0
            data['ca_2'] = 0
            data['ca_3'] = 0
            data['ca_4'] = 0
        elif int(request.form['ca'])==1:
            data['ca_0'] = 0
            data['ca_1'] = 1
            data['ca_2'] = 0
            data['ca_3'] = 0
            data['ca_4'] = 0
        elif int(request.form['ca'])==2:
            data['ca_0'] = 0
            data['ca_1'] = 0
            data['ca_2'] = 1
            data['ca_3'] = 0
            data['ca_4'] = 0
        elif int(request.form['ca'])==3:
            data['ca_0'] = 0
            data['ca_1'] = 0
            data['ca_2'] = 0
            data['ca_3'] = 1
            data['ca_4'] = 0
        else:
            data['ca_0'] = 0
            data['ca_1'] = 0
            data['ca_2'] = 0
            data['ca_3'] = 0
            data['ca_4'] = 1
            
        if int(request.form['thal']) == 0:
            data['thal_0'] = 1
            data['thal_1'] = 0
            data['thal_2'] = 0
            data['thal_3'] = 0
        elif int(request.form['thal']) == 1:
            data['thal_0'] = 0
            data['thal_1'] = 1
            data['thal_2'] = 0
            data['thal_3'] = 0
        elif int(request.form['thal']) == 2:
            data['thal_0'] = 0
            data['thal_1'] = 0
            data['thal_2'] = 1
            data['thal_3'] = 0
        else:
            data['thal_0'] = 0
            data['thal_1'] = 0
            data['thal_2'] = 0
            data['thal_3'] = 1
        df = pd.DataFrame(data,index=[0])
        

        #converted_df = pd.get_dummies(df,columns=to_get_dummy)
        df[to_scale]=scaler.fit_transform(df[to_scale])
        
        sample = df.to_numpy()
        
        prediction = model.predict(sample.reshape(1,-1))
        prediction = prediction[0]
        
        if prediction == 1:
            return render_template('index.html',prediction_text="Prediction Result:Positive")
        else:
            return render_template('index.html',prediction_text="Prediction Result:negative")
    
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
        
        
        
