import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

st.set_page_config(layout="wide")
st.markdown(
            """
            <style>
            [data-testid="stElementToolbar"] {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# st.title("WGI Predictions App")
# st.write("This app shows recaps for Bands of America events from the 2024 season. Use the drop down selectors to filter by event or show round, or leave them empty to view all scores. Click on a column header to cycle through sorting options. When hovering over a column header, click the hamburger menu on the right to view more filtering options. \n")

classes = ['Independent A', 'Scholastic A', 'Independent Open', 'Scholastic Open',
           'Independent World', 'Scholastic World']

row_input = st.columns((1,1,1))
with row_input[0]:
    class_25 = st.selectbox('What class is the guard competing in during the 2025 season?', classes,
                            placeholder='Independent World', index=None)
    round = st.radio("What show round did the score that you're entering occur in?", ['Prelims','Finals'])

with row_input[1]:
    week = st.selectbox("What show week did the score that you're entering occur in?", 
                      ['Week 1: 2/8-9','Week 2: 2/15-16', 'Week 3: 2/22-23', 'Week 4: 3/1-2',
                       'Week 5: 3/8-9', 'Week 6: 3/15-16', 'Week 7: 3/22-23'], 
                      placeholder='Week 1: 2/8-9', index=None)
    choice = st.radio('Do you want to enter caption scores?', ['Yes', 'No'])

row_input = st.columns((1,1,1))
if choice == 'Yes':
    with row_input[0]:
        ea_tot_sc = st.number_input("Equipment Analysis Total Score (out of 20 points)", min_value=0.0, max_value=20.0, format="%0.3f")
        da_tot_sc = st.number_input("Design Analysis Total Score (out of 20 points)", min_value=0.0, max_value=20.0, format="%0.3f")
    with row_input[1]:
        ma_tot_sc = st.number_input("Movement Analysis Total Score (out of 20 points)", min_value=0.0, max_value=20.0, format="%0.3f")
        tot_ge_sc = st.number_input("Total GE Score (out of 40 points)", min_value=0.0, max_value=40.0, format="%0.3f")
    subtot_sc = ea_tot_sc + ma_tot_sc + da_tot_sc + tot_ge_sc
    st.write('Total Subtotal score: ', "{:.3f}".format(subtot_sc))
    
if choice == 'No':    
    with row_input[0]:
        subtot_sc = st.number_input("Subtotal Score", min_value=0.0, max_value=100.0, format="%0.3f")
    ea_tot_sc = subtot_sc * 0.2
    ma_tot_sc = subtot_sc * 0.2
    da_tot_sc = subtot_sc * 0.2
    tot_ge_sc = subtot_sc * 0.4

if week == 'Week 1: 2/8-9':
    seed = subtot_sc + 9.0
if week == 'Week 2: 2/15-16':
    seed = subtot_sc + 7.5
if week == 'Week 3: 2/22-23':
    seed = subtot_sc + 6.0
if week == 'Week 4: 3/1-2':
    seed = subtot_sc + 4.5
if week == 'Week 5: 3/8-9':
    seed = subtot_sc + 3.0
if week == 'Week 6: 3/15-16':
    seed = subtot_sc + 1.5
if week == 'Week 7: 3/22-23':
    seed = subtot_sc

row_input = st.columns((1,1,1))
with row_input[0]:
    class_24 = st.selectbox('What class did the guard compete in during the 2024 season? If the guard did not compete,select the class that guard is in during the 2025 season', classes, placeholder='Independent World', index=None)
with row_input[0]:
    prv_year = st.radio('Did the guard compete at the 2024 World Championships?', ['Yes', 'No'])

if prv_year == 'No':
    prv_round_num = 0
    prv_score = 0
    prv_place = 0
if prv_year == 'Yes':
    with row_input[0]:
        prv_round = st.radio('What is the LAST round the guard competed in at the 2024 World Championships?', ['Prelims', 'Semifinals', 'Finals'])
    with row_input[0]:
        prv_score = st.number_input('What score did the guard receive in their LAST round at the 2024 World Championships?', min_value=0.0, max_value=100.0, format="%0.3f")
    row_input = st.columns((1,1))
    with row_input[0]:
        prv_place = st.number_input("What OVERALL placement did the guard receive in their LAST round at the 2024 World Championships? If the last round was semifinals or prelims, this should be the guard's overall placement in the entire class, NOT placement within their round. Overall placements can be found using our WGI Historic Scores page at https://bandscores.net/wgi-guard-historic", min_value=0, max_value=140, step=1)
    if prv_round == 'Prelims':
        prv_round_num = 1
    if prv_round == 'Semifinals':
        prv_round_num = 2
    if prv_round == 'Finals':
        prv_round_num = 3

if round == 'Prelims':
    round_num = 1
if round == 'Finals':
    round_num = 3

if class_25 is not None:
    class_25_num = classes.index(class_25)+1
if class_24 is not None:
    class_24_num = classes.index(class_24)+1


# Predict button
if st.button("Predict Final Score"):
    data = pd.read_csv('wgi_train.csv')
    model_data = data[['ID', 'Round_Numb','Class Numb',
                      'EA_Tot_Sc', 'MA_Tot_Sc', 'DA_Tot_Sc', 'Tot_GE_Sc',
                      'Subtot_Sc', 'Seed Score',
                      'Prv Class', 'Prv WC Round', 'Prv Fin Score', 'Prv Fin Place',
                      'Fin Score']]
    
    model_data = model_data.dropna()
    X = model_data.iloc[:,1:-1]
    y = model_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=15)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set the parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.2,
        'n_estimators': 100,
        'verbosity': 0
    }
    
    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    fin_data = data[['ID', 'Round_Numb','Class Numb',
                      'EA_Tot_Sc', 'MA_Tot_Sc', 'DA_Tot_Sc', 'Tot_GE_Sc',
                      'Subtot_Sc', 'Seed Score',
                      'Prv Class', 'Prv WC Round', 'Prv Fin Score', 'Prv Fin Place',
                      'Finalist']]
    
    fin_data = fin_data.dropna()
    X_fin = fin_data.iloc[:,1:-1]
    y_fin = fin_data.iloc[:, -1]
    X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X_fin, y_fin, test_size=0.05, random_state=15)
    
    dtrain_fin = xgb.DMatrix(X_train_fin, label=y_train_fin)
    dtest_fin = xgb.DMatrix(X_test_fin, label=y_test_fin)
    
    # Set the parameters for XGBoost
    fin_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.2,
        'n_estimators': 100,
        'verbosity': 0
    }
    
    finalist_model = xgb.train(params, dtrain_fin, num_boost_round=100)
  
    input = pd.DataFrame([{'Round_Numb':round_num ,'Class Numb':class_25_num,
                          'EA_Tot_Sc':ea_tot_sc, 'MA_Tot_Sc':ma_tot_sc, 'DA_Tot_Sc':da_tot_sc, 
                          'Tot_GE_Sc':tot_ge_sc, 'Subtot_Sc':subtot_sc, 'Seed Score':seed, 
                          'Prv Class':class_24_num, 'Prv WC Round':prv_round_num, 
                          'Prv Fin Score':prv_score, 'Prv Fin Place':prv_place}])
        
    dinput = xgb.DMatrix(input)
    prediction = min(model.predict(dinput)[0],100) if model else "Model not available"

    finalist_prob = finalist_model.predict(dinput)[0] if finalist_model else "Model not available"
    finalist_percentage = min((finalist_prob * 100),100) if finalist_model else "Error"
       
    st.subheader(f"Predicted Final Score at World Championships: {prediction:.3f}" if model else "Error: Model not loaded.")
    st.subheader(f"Odds of Making Finals at World Championships: {finalist_percentage:.2f}%" if finalist_model else "Error: Model not loaded.")



