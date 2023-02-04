import pickle
from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open("gbr.pkl","rb"))
scalar=pickle.load(open("scaling.pkl","rb"))
@app.route('/')
def home():
    return render_template("index.html")
@app.route("/predict", methods=['POST'])
def predict():
    if request.method=="POST":
        cement=float(request.form.get("cement"))
        slag=float(request.form.get("slag"))
        flyash=float(request.form.get("flyash"))
        water=float(request.form.get("water"))
        superplasticizer=float(request.form.get("superplasticizer"))
        coarseaggregate=float(request.form.get("coarseaggregate"))    
        fineaggregate=float(request.form.get("fineaggregate")) 
        age =int(request.form.get("age"))
        X=scalar.transform([[cement,slag,flyash,water,superplasticizer,coarseaggregate,fineaggregate,age]])
        prediction=model.predict(X)
        output=round(prediction[0],2)
        if output<=0:
            return render_template('index.html',prediction_texts="incorrect values")
        else:
            return render_template('index.html',prediction_text="the compressive  strength in {} Mpa".format(output))       
    else:
       return render_template("index.html")     

if __name__=="__main__":
    app.run(debug=True)



