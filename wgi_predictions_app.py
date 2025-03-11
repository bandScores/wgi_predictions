import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

# Hide Streamlit Toolbar
st.markdown(
    """
    <style>
    [data-testid="stElementToolbar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    data = pd.read_csv('wgi_train.csv')
    return data.dropna()

@st.cache_resource
def train_models(data):
    # Prepare data for final score prediction
    model_data = data[['Round_Numb', 'Class Numb', 'EA_Tot_Sc', 'MA_Tot_Sc', 'DA_Tot_Sc', 
                       'Tot_GE_Sc', 'Subtot_Sc', 'Seed Score', 'Prv Class', 
                       'Prv WC Round', 'Prv Fin Score', 'Prv Fin Place', 'Fin Score']]
    
    X = model_data.iloc[:, :-1]
    y = model_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=15)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.2, 'n_estimators': 100, 'verbosity': 0}
    score_model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Prepare data for finalist prediction
    fin_data = data[['Round_Numb', 'Class Numb', 'EA_Tot_Sc', 'MA_Tot_Sc', 'DA_Tot_Sc', 
                     'Tot_GE_Sc', 'Subtot_Sc', 'Seed Score', 'Prv Class', 
                     'Prv WC Round', 'Prv Fin Score', 'Prv Fin Place', 'Finalist']]
    
    X_fin = fin_data.iloc[:, :-1]
    y_fin = fin_data.iloc[:, -1]
    X_train_fin, X_test_fin, y_train_fin, y_test_fin = train_test_split(X_fin, y_fin, test_size=0.05, random_state=15)
    
    dtrain_fin = xgb.DMatrix(X_train_fin, label=y_train_fin)
    finalist_model = xgb.train(params, dtrain_fin, num_boost_round=100)

    return score_model, finalist_model

# Load data and train models once
data = load_data()
score_model, finalist_model = train_models(data)

# UI Inputs
classes = ['Independent A', 'Scholastic A', 'Independent Open', 'Scholastic Open', 'Independent World', 'Scholastic World']

with st.sidebar:
    class_25 = st.selectbox('2025 Guard Class', classes, key='class_25')
    round_num = 1 if st.radio("Show Round", ['Prelims', 'Finals'], key='round') == 'Prelims' else 3
    week = st.selectbox("Show Week", [f"Week {i}: {date}" for i, date in enumerate(["2/8-9", "2/15-16", "2/22-23", "3/1-2", "3/8-9", "3/15-16", "3/22-23"], 1)], key='week')
    caption_choice = st.radio('Enter caption scores?', ['Yes', 'No'], key='caption')

# Caption Inputs
ea_tot_sc, ma_tot_sc, da_tot_sc, tot_ge_sc = 0, 0, 0, 0
if caption_choice == 'Yes':
    ea_tot_sc = st.number_input("Equipment Analysis Total Score", min_value=0.0, max_value=20.0, format="%0.3f")
    ma_tot_sc = st.number_input("Movement Analysis Total Score", min_value=0.0, max_value=20.0, format="%0.3f")
    da_tot_sc = st.number_input("Design Analysis Total Score", min_value=0.0, max_value=20.0, format="%0.3f")
    tot_ge_sc = st.number_input("Total GE Score", min_value=0.0, max_value=40.0, format="%0.3f")
    subtot_sc = ea_tot_sc + ma_tot_sc + da_tot_sc + tot_ge_sc
else:
    subtot_sc = st.number_input("Subtotal Score", min_value=0.0, max_value=100.0, format="%0.3f")

st.write(f'Total Subtotal score: {subtot_sc:.3f}')

# Calculate Seed Score
week_offsets = [9.0, 7.5, 6.0, 4.5, 3.0, 1.5, 0.0]
seed = subtot_sc + week_offsets[int(week.split()[1]) - 1]

# Previous Year Inputs
class_24 = st.selectbox('2024 Guard Class', classes, key='class_24')
prv_year = st.radio('Did the guard compete at 2024 World Championships?', ['Yes', 'No'], key='prv_year')

prv_round_num, prv_score, prv_place = 0, 0, 0
if prv_year == 'Yes':
    prv_round_map = {'Prelims': 1, 'Semifinals': 2, 'Finals': 3}
    prv_round = st.radio('Last round in 2024 Championships?', list(prv_round_map.keys()), key='prv_round')
    prv_round_num = prv_round_map[prv_round]
    prv_score = st.number_input('Last Score at 2024 Championships?', min_value=0.0, max_value=100.0, format="%0.3f")
    prv_place = st.number_input("Overall Placement in Last Round", min_value=0, max_value=140, step=1)

# Convert classes to numerical
class_25_num = classes.index(class_25) + 1 if class_25 else None
class_24_num = classes.index(class_24) + 1 if class_24 else None

# Predict button
if st.button("Predict Final Score"):
    input_data = pd.DataFrame([{
        'Round_Numb': round_num, 'Class Numb': class_25_num,
        'EA_Tot_Sc': ea_tot_sc, 'MA_Tot_Sc': ma_tot_sc, 'DA_Tot_Sc': da_tot_sc,
        'Tot_GE_Sc': tot_ge_sc, 'Subtot_Sc': subtot_sc, 'Seed Score': seed,
        'Prv Class': class_24_num, 'Prv WC Round': prv_round_num,
        'Prv Fin Score': prv_score, 'Prv Fin Place': prv_place
    }])

    dinput = xgb.DMatrix(input_data)
    
    predicted_score = min(score_model.predict(dinput)[0], 100)
    finalist_prob = min(finalist_model.predict(dinput)[0] * 100, 100)

    st.subheader(f"Predicted Final Score: {predicted_score:.3f}")
    st.subheader(f"Odds of Making Finals: {finalist_prob:.2f}%")
