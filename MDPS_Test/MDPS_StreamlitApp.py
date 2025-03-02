import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from MDPS_Preprocess import preprocess_data
from streamlit_option_menu import option_menu

# Load pre-trained models and scalers
model_liver = joblib.load('models/liver_model.pkl')
model_kidney = joblib.load('models/kidney_model.pkl')
model_parkinson = joblib.load('models/parkinson_model.pkl')

scaler_liver = joblib.load('models/scaler_liver.pkl')
scaler_kidney = joblib.load('models/scaler_kidney.pkl')
scaler_parkinson = joblib.load('models/scaler_parkinson.pkl')

#st.header("Multiple Disease Prediction: Liver, Kidney, Parkinson's Disease")

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',                           
                           ['Kidney Prediction',
                            'Liver Prediction',
                            'Parkinsons Prediction'],
                            icons=['Kidney','heart','house'],
                            menu_icon='cast',                            
                            default_index=0)
if selected == 'Parkinsons Prediction':
    st.header("Parkinson's Disease Prediction")

    # Input fields for each disease (liver, kidney, parkinson)
    MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
    MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
    MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0)
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0)
    MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.0)
    MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.0)
    Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.0)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0)
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0)
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0)
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0)
    MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0)
    Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.0)
    NHR = st.number_input("NHR", min_value=0.0)
    HNR = st.number_input("HNR", min_value=0.0)
    RPDE = st.number_input("RPDE", min_value=0.0)    
    DFA = st.number_input("DFA", min_value=0.0)
    spread1 = st.number_input("spread1", min_value=0.0)
    spread2 = st.number_input("spread2", min_value=0.0)
    D2 = st.number_input("D2", min_value=0.0)
    PPE = st.number_input("PPE", min_value=0.0)
   
    # Add more fields as per the disease (Liver, Kidney, Parkinson)

    # Creating input dataframe for prediction
    input_data = pd.DataFrame([[MDVP_Fo,  MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                                MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, 
                                RPDE, DFA, spread1, spread2, D2, PPE]], columns=["MDVP_Fo",  "MDVP_Fhi", "MDVP_Flo", "MDVP_Jitter", "MDVP_Jitter_Abs", "MDVP_RAP", "MDVP_PPQ", "Jitter_DDP",
                                "MDVP_Shimmer", "MDVP_Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "MDVP_APQ", "Shimmer_DDA", "NHR", "HNR", 
                                "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"])

    # Preprocessing for liver, kidney, and Parkinson's data
    input_data_parkinson, scaler_parkinson = preprocess_data(input_data, 'parkinson', scaler_parkinson)

    # Make predictions for each disease
    pred_parkinson = model_parkinson.predict(input_data_parkinson)

    # Show predictions
    st.write(f"Parkinson's Disease Prediction: {'Yes' if pred_parkinson == 1 else 'No'}")

    # Optional: Visualize probabilities (bar charts, pie charts, etc.)

elif selected == 'Liver Prediction':
    st.header("Liver Disease Prediction")

    # Input fields for each disease (liver, kidney, parkinson)
    # Similar to the current input for liver disease, repeat for kidney and Parkinson's with necessary inputs
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Age = st.number_input("Age", min_value=0)
    
    # Common for all diseases (example input, can be expanded)
    Total_Bilirubin = st.number_input("Total_Bilirubin")
    # Add more fields as per the disease (Liver, Kidney, Parkinson)
    Direct_Bilirubin = st.number_input("Direct_Bilirubin")
    Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase")
    Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase")
    Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase")
    Total_Protiens = st.number_input("Total_Protiens")
    Albumin = st.number_input("Albumin")
    Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio")

    # Creating input dataframe for prediction
    input_data = pd.DataFrame([[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase,
                                Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]], 
                                columns=[ "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", 
                                         "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", 
                                         "Albumin", "Albumin_and_Globulin_Ratio"])

    # Preprocessing for liver, kidney, and Parkinson's data
    input_data_liver, scaler_liver = preprocess_data(input_data, 'liver', scaler_liver)
    

    # Make predictions for each disease
    pred_liver = model_liver.predict(input_data_liver)
    

    # Show predictions
    st.write(f"Liver Disease Prediction: {'Yes' if pred_liver == 1 else 'No'}")
    

    # Optional: Visualize probabilities (bar charts, pie charts, etc.)
elif selected == 'Kidney Prediction':
    st.header("Kidney Disease Prediction")

    # Input fields for each disease (liver, kidney, parkinson)
    # Similar to the current input for liver disease, repeat for kidney and Parkinson's with necessary inputs
    age = st.number_input("age", min_value=0)
    bp = st.number_input("bp", min_value=0)
    sg = st.number_input("sg", min_value=0.0)
    al = st.number_input("al", min_value=0)
    su = st.number_input("su", min_value=0)
    rbc = st.selectbox("rbc", ['normal', 'abnormal'])
    pc = st.selectbox("pc", ['normal', 'abnormal'])
    pcc = st.selectbox("pcc", ['present', 'notpresent'])
    ba = st.selectbox("ba", ['present', 'notpresent'])
    bgr = st.number_input("bgr", min_value=0)
    bu = st.number_input("bu", min_value=0)
    sc = st.number_input("sc", min_value=0.0)
    sod = st.number_input("sod", min_value=0)
    pot = st.number_input("pot", min_value=0.0)
    hemo = st.number_input("hemo", min_value=0.0)
    pcv = st.number_input("pcv", min_value=0)
    wc = st.number_input("wc", min_value=0)
    rc = st.number_input("rc", min_value=0.0)
    htn = st.selectbox("htn", ['yes', 'no'])
    dm = st.selectbox("dm", ['yes', 'no'])
    cad = st.selectbox("cad", ['yes', 'no'])
    appet = st.selectbox("appet", ['good', 'poor'])
    pe = st.selectbox("pe", ['yes', 'no'])
    ane = st.selectbox("ane", ['yes', 'no'])
    classification = st.selectbox("classification", ['ckd', 'notckd'])
    

    # Creating input dataframe for prediction
    input_data = pd.DataFrame([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, classification]], columns=["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane", "classification"])

    # Preprocessing for liver, kidney, and Parkinson's data
   
    input_data_kidney, scaler_kidney = preprocess_data(input_data, 'kidney', scaler_kidney)
    

    # Make predictions for each disease
    pred_kidney = model_kidney.predict(input_data_kidney)
    

    # Show predictions    
    st.write(f"Kidney Disease Prediction: {'Yes' if pred_kidney == 1 else 'No'}")
    

    # Optional: Visualize probabilities (bar charts, pie charts, etc.)


