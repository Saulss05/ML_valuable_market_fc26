import streamlit as st
import numpy as np
import pandas as pd
import joblib

#Chargement des données
model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")
features=joblib.load("features.pkl")
positions_cols=joblib.load("positions_cols.pkl")
nations_cols=joblib.load("nations_cols.pkl")
leagues_names_cols=joblib.load("leagues_names_cols.pkl")

#Titre du modèle
st.title("EA FC 26 market value prediction model")

#Inputs numériques
overall=st.slider("Overall",40,99,75)
potential=st.slider("Potential",40,99,80)
age=st.slider("Age",16,45,25)

#Input choix
position=st.selectbox("Position",positions_cols)
nation = st.selectbox("Nation", ["Other"] + [c.replace("nat_", "") for c in nations_cols])
league = st.selectbox("League", ["Other"] + [c.replace("lg_name_", "") for c in leagues_names_cols])

if st.button("Prédire la valeur"):

    #Création ligne vide
    input_dict = {col: 0 for col in features}

    # Numériques
    input_dict["overall_exp"] = np.exp(overall / 30)
    input_dict["potential"] = potential
    input_dict["age"] = age

    #Position
    if position in positions_cols:
        input_dict[position] = 1
    
    #Nation
    nat_col = f"nat_{nation}"
    if nat_col in nations_cols:
        input_dict[nat_col] = 1

    #League
    lg_col=f"lg_name_{league}"
    if lg_col in leagues_names_cols:
        input_dict[lg_col]=1

    # DataFrame
    input_df = pd.DataFrame([input_dict])

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction log
    pred_log = model.predict(input_scaled)

    # Retour valeur réelle
    pred_value = np.exp(pred_log)[0]

    st.success(f"Valeur estimée : {int(pred_value):,} €")

