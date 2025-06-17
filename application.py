import pickle
import pandas as pd
import numpy as np
from flask import Flask, jsonify,request,render_template
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pickle','rb'))
scaler_model=pickle.load(open('models/scaler.pickle','rb'))
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Class=float(request.form.get('Class'))
        Region=float(request.form.get('Region'))
        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Class,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('predict.html',result=result[0])
    else:
        return render_template('predict.html')
if __name__=="__main__":
    app.run(host="0.0.0.0")