import pickle
from flask import Flask, redirect, url_for, render_template,request
model_path = 'models/'
model_limit = pickle.load(open(model_path+'model_limit.pkl', 'rb'))
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


with open(model_path+'pca_model_crab.pkl', 'rb') as file:
    pca_crab = pickle.load(file)
with open(model_path+'pca_model_delfin.pkl', 'rb') as file:
    pca_delfin = pickle.load(file)
with open(model_path+'pca_model_pinguin.pkl', 'rb') as file:
    pca_pinguin = pickle.load(file)
with open(model_path+'model_crab.pkl','rb') as file:
    model_crab=pickle.load(file)
with open(model_path + 'model_delfin.pkl', 'rb') as file:
    model_delfin = pickle.load(file)
with open(model_path + 'model_pinguin.pkl', 'rb') as file:
    model_pinguin = pickle.load(file)


app = Flask(__name__)
vector=[]
credit_limit=0
offer=0
crab=0
pinguin=0
delfin=0
produs=''
@app.route('/')
def home():  # put application's code here
    global vector,credit_limit,offer,crab,pinguin,delfin,produs
    vector = []
    credit_limit = 0
    offer = 0
    crab = 0
    pinguin = 0
    delfin = 0
    produs=''
    result2=None
    result=None
    return render_template('index.html')

@app.route('/analyze', methods=["GET","POST"])
def analyze():
    global crab,pinguin,delfin
    if 'button' in request.form:
        button_value = request.form['button']
        if button_value == 'crab':
            delfin = 0
            crab = 1
            pinguin = 0
            classname="backgrnd_crab"
            produs="Crab"
            #return render_template('form_crab.html',classname=classname,produs=produs)
        if button_value == 'delfin':
            delfin=1
            crab=0
            pinguin=0
            classname="backgrnd_delfin"
            produs = "Delfin"
            #return render_template('form_delfin.html',classname=classname,produs=produs)
        if button_value == 'pinguin':
            delfin = 0
            crab = 0
            pinguin = 1
            classname="backgrnd_pinguin"
            produs = "Pinguin"
        return render_template('form_crab.html',classname=classname,produs=produs)

@app.route('/result', methods=["GET","POST"])
def result():

    limit_vector=[2]
    if request.method=='POST':
        global vector,credit_limit,crab,delfin,pinguin,produs
        data = request.form
        phone=request.form['phone']
        nume=(request.form['firstname'])
        prenume = (request.form['lastname'])
        varsta = request.form['varsta']
        gender = request.form["gender"]
        income = request.form['income']
        score = int(request.form['score'])
        other_credits = int(request.form['other_credits'])
        credits_before = int(request.form['credits_before'])
        bnr40=int(0.4* int(income) - int(other_credits))
        limit_vector=[gender,varsta,score,income,other_credits,bnr40,credits_before]
        vector.extend(limit_vector)
        if crab==1:
            limit_vector.extend([1,0,0])
        elif pinguin==1:
            limit_vector.extend([0,0,1])
        elif delfin ==1:
            limit_vector.extend([0,1,0])
        limit_series=pd.Series(limit_vector)
        credit_limit=int(model_limit.predict(limit_series.values.reshape(1,-1)))
        return render_template('oferta.html',credit_lim=credit_limit,produs=produs)

    return render_template('oferta.html')

@app.route('/prediction', methods=["POST","GET"])
def prediction():
    if request.method=='POST':
        global vector,credit_limit,offer,crab,delfin,pinguin,produs,result,result2
        #prod_list=[crab,delfin,pinguin]
        offer=request.form['loan-amount']
        vector.insert(2,credit_limit)
        vector.append(offer)
        #vector.extend(offer_list)
        #vector.extend(prod_list)
        vector=np.array(vector).reshape(1, -1)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        vector_rescaled = min_max_scaler.fit_transform(vector)
        if crab==1:
            transformed_vector=pca_crab.transform(vector_rescaled)
            result=model_crab.predict(transformed_vector)
            result2 = model_crab.predict_proba(transformed_vector)
        elif pinguin==1:
            transformed_vector =(pca_pinguin.transform(vector_rescaled))
            result = model_pinguin.predict(transformed_vector)
            result2 = model_pinguin.predict_proba(transformed_vector)
        elif delfin==1:
            transformed_vector = pca_delfin.transform(vector_rescaled)
            result = model_delfin.predict(transformed_vector)
            result2 = model_delfin.predict_proba(transformed_vector)
        if int(result[0])==0:
            result_text="Nu prezinta risc"
            class_name='wrapper_good'
            result2=result2[0]*100
        elif int(result[0])==1:
            result_text = "Prezinta risc"
            class_name="wrapper_bad"
            result2=result2[1]*100



        return render_template('prediction.html',result=result_text,result2=result2,produs=produs,class_name=class_name)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)


