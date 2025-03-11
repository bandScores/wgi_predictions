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

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}

def update_inputs(key, value):
    st.session_state.user_inputs[key] = value

# UI Inputs
classes = ['Independent A', 'Scholastic A', 'Independent Open', 'Scholastic Open', 'Independent World', 'Scholastic World']

col1, col2, col3 = st.columns((1, 1, 1))
with col1:
    class_25 = st.selectbox("2025 Class", classes, index=None, on_change=update_inputs, args=("class_25", class_25))
    round_2025 = st.radio("Round", ["Prelims", "Finals"], on_change=update_inputs, args=("round", round_2025))

with col2:
    week = st.selectbox("Show Week", weeks, index=None, on_change=update_inputs, args=("week", week))
    captions = st.radio("Enter Caption Scores?", ["Yes", "No"], on_change=update_inputs, args=("captions", captions))

# Caption Inputs (only show if needed)
if st.session_state.user_inputs.get("captions") == "Yes":
    with col1:
        ea_tot_sc = st.number_input("EA Score", min_value=0.0, max_value=20.0, format="%0.2f")
        da_tot_sc = st.number_input("DA Score", min_value=0.0, max_value=20.0, format="%0.2f")
    with col2:
        ma_tot_sc = st.number_input("MA Score", min_value=0.0, max_value=20.0, format="%0.2f")
        tot_ge_sc = st.number_input("GE Score", min_value=0.0, max_value=40.0, format="%0.2f")
    subtot_sc = ea_tot_sc + ma_tot_sc + da_tot_sc + tot_ge_sc
else:
    with col1:
        subtot_sc = st.number_input("Subtotal Score", min_value=0.0, max_value=100.0, format="%0.2f")
    ea_tot_sc = subtot_sc * 0.2
    ma_tot_sc = subtot_sc * 0.2
    da_tot_sc = subtot_sc * 0.2
    tot_ge_sc = subtot_sc * 0.4

# Calculate Seed Score
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

st.write(f'Total Score that you entered: {subtot_sc:.3f}')
#st.write('Seeding score based off subtotal score:', seed)

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
