
import pandas as pd 
import numpy as np 
from tensorflow.keras.models import load_model 
import streamlit as st

import pickle


Regerssion_model=load_model("Regerssion_model.h5",compile=False
)

with open("Gender_encoder_regression.pkl","rb") as file:
    Gender_encoder_regression=pickle.load(file)

###Geography
with open("OHE_geography_regression.pkl","rb") as file:
    OHE_geography_regression=pickle.load(file)
##SCaling
with open("scaler_regression.pkl","rb") as file:
    scaler_regression=pickle.load(file)




st.title("Customer Salary Prediction")



georaphy=st.selectbox("Georaphy",OHE_geography_regression.categories_[0])
gender=st.selectbox("Gender",Gender_encoder_regression.classes_)
age=st.slider("Age",18,95)
CreditScore=st.number_input("CreditScore")
balance=st.number_input("Balance")
Exited=st.selectbox("Has Exited",[0,1])
tenure=st.slider("Tenure",0,10)
num_of_products=st.selectbox("Number of products",[1,2,3,4])
has_cr_card=st.selectbox("has credict card ",[0,1])
is_active_member=st.selectbox("Is active member",[0,1])

input_data=pd.DataFrame({
    
    "CreditScore":[CreditScore],
    "Geography"	:[georaphy],
    "Gender":[Gender_encoder_regression.transform([gender])[0]],
    "Age":[age],	
    "Tenure":[tenure],	
    "Balance":[balance],	
    "NumOfProducts":[num_of_products],	
    "HasCrCard":[has_cr_card],	
    "IsActiveMember":[is_active_member],	
    "Exited":[Exited]
})

geo_encoded=OHE_geography_regression.transform([[georaphy]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=OHE_geography_regression.get_feature_names_out(["Geography"]))









input_data=pd.concat([input_data.drop("Geography",axis=1),geo_encoded_df],axis=1)

input_data_scaled=scaler_regression.transform(input_data)


prediction=Regerssion_model.predict(input_data_scaled)



st.write(f"The Esimated Salary of the coustomer{prediction[0][0]}")
