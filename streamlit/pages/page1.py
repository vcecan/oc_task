import streamlit as st
import os
import sys


from prediction import predict
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.set_page_config(
    page_title="Form",
    page_icon="📝"
)


age = st.number_input("Introduceti varsta", 0, 100)
gender = st.select_slider('Alegeti Genul', ["Barbat", "Femeie"])
if gender=='Barbat':
    gender=1
elif gender=='Femeie':
    gender=0
produs = st.selectbox("Alegeti tipul de produs", ['Crab', 'Delfin', 'Pinguin'])
credit_limit=0
if produs=="Crab":
    credit_limit = st.number_input('Limita creditara:', 0, 1500)
elif produs=="Delfin":
    credit_limit = st.number_input('Limita creditara:', 0, 9000)
elif produs == "Pinguin":
    credit_limit = st.number_input('Limita creditara:', 0, 4500)

score = st.number_input('Scorul Bancar(In caz ca nu aveti,introduceti -1):', -1, 850)
income = st.number_input('Venitul Lunar:', 0, 200000)
other_credits = st.number_input('Plata totala a altor credite', 0, 100000)
bnr40 = income*0.4-other_credits
offer_crab=0
offer_delfin=0
offer_pinguin=0
if produs=="Crab":
    offer_crab=credit_limit
elif produs=="Delfin":
    offer_delfin=credit_limit
elif produs == "Pinguin":
    offer_pinguin = credit_limit/3

comission = st.selectbox("Alegeti Comissionul", ['0','5','7','9'])


def result():
    result = predict(age, gender, score, credit_limit, income, bnr40, offer_crab, offer_delfin, offer_pinguin, produs,
                     other_credits, comission)

    st.write("Model Result:", result)


st.button('Predict', on_click=result)