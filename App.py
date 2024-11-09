   import streamlit as st
   import pandas as pd
   import numpy as np
   import pickle
   from sklearn.preprocessing import StandardScaler

   # Load the trained model
   with open('model.pkl', 'rb') as model_file:
       model = pickle.load(model_file)

   # Streamlit app title
   st.title("Liver Disease Prediction")

   # Sidebar for user input
   st.sidebar.header("Enter Patient Data:")

   # Function to take user input
   def user_input():
       age = st.sidebar.slider("Age", 0, 100, 50)
       sex = st.sidebar.selectbox("Sex", ("m", "f"))
       albumin = st.sidebar.number_input("Albumin (g/L)", min_value=0.0, max_value=100.0, value=44.0)
       alkaline_phosphatase = st.sidebar.number_input("Alkaline Phosphatase (U/L)", min_value=0.0, max_value=500.0, value=60.0)
       alanine_aminotransferase = st.sidebar.number_input("Alanine Aminotransferase (U/L)", min_value=0.0, max_value=1000.0, value=40.0)
       aspartate_aminotransferase = st.sidebar.number_input("Aspartate Aminotransferase (U/L)", min_value=0.0, max_value=1000.0, value=30.0)
       bilirubin = st.sidebar.number_input("Bilirubin (mg/L)", min_value=0.0, max_value=50.0, value=5.0)
       cholinesterase = st.sidebar.number_input("Cholinesterase (U/L)", min_value=0.0, max_value=30.0, value=10.0)
       cholesterol = st.sidebar.number_input("Cholesterol (mmol/L)", min_value=0.0, max_value=10.0, value=4.5)
       creatinine = st.sidebar.number_input("Creatinine (μmol/L)", min_value=0.0, max_value=200.0, value=80.0)
       gamma_glutamyl_transferase = st.sidebar.number_input("Gamma-Glutamyl Transferase (IU/L)", min_value=0.0, max_value=100.0, value=25.0)
       protein = st.sidebar.number_input("Protein (mg)", min_value=0.0, max_value=100.0, value=70.0)
       
       # Create a dictionary with user input
       user_data = {
           "age": age,
           "sex": 1 if sex == 'm' else 0,  # Binary encoding for sex
           "albumin": albumin,
           "alkaline_phosphatase": alkaline_phosphatase,
           "alanine_aminotransferase": alanine_aminotransferase,
           "aspartate_aminotransferase": aspartate_aminotransferase,
           "bilirubin": bilirubin,
           "cholinesterase": cholinesterase,
           "cholesterol": cholesterol,
           "creatinine": creatinine,
           "gamma_glutamyl_transferase": gamma_glutamyl_transferase,
           "protein": protein
       }
       return pd.DataFrame(user_data, index=[0])

   # Display user input
   input_df = user_input()
   st.write("## Patient Data:", input_df)

   # Make predictions and display results
   if st.button("Predict"):
       prediction = model.predict(input_df)
       prediction_proba = model.predict_proba(input_df)

       # Map numerical prediction to disease category
       disease_map = {0: "No disease", 1: "Suspect disease", 2: "Hepatitis C", 3: "Fibrosis", 4: "Cirrhosis"}
       st.write(f"## Prediction: {disease_map[prediction[0]]}")
       st.write("## Prediction Probabilities:")
       for i, prob in enumerate(prediction_proba[0]):
           st.write(f"{disease_map[i]}: {prob:.2f}")
   
