import streamlit as st
import pandas as pd
import xgboost as xgb

# --- Set Page Layout ---
st.set_page_config(layout="wide")

# --- Load Data & Models ---
@st.cache_data
def load_data():
    return pd.read_csv('wgi_train.csv')

@st.cache_resource
def load_models():
    data = load_data()

    # Regression Model (Predicts Final Score)
    model_data = data[['Round_Numb', 'Class Numb', 'EA_Tot_Sc', 'MA_Tot_Sc', 
                       'DA_Tot_Sc', 'Tot_GE_Sc', 'Subtot_Sc', 'Seed Score', 
                       'Prv Class', 'Prv WC Round', 'Prv Fin Score', 'Prv Fin Place', 
                       'Fin Score']].dropna()
    
    X_train = model_data.iloc[:, :-1]
    y_train = model_data.iloc[:, -1]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.2}
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Classification Model (Predicts Odds of Making Finals)
    fin_data = data[['Round_Numb', 'Class Numb', 'EA_Tot_Sc', 'MA_Tot_Sc', 
                     'DA_Tot_Sc', 'Tot_GE_Sc', 'Subtot_Sc', 'Seed Score', 
                     'Prv Class', 'Prv WC Round', 'Prv Fin Score', 'Prv Fin Place', 
                     'Finalist']].dropna()
    
    X_train_fin = fin_data.iloc[:, :-1]
    y_train_fin = fin_data.iloc[:, -1]
    
    dtrain_fin = xgb.DMatrix(X_train_fin, label=y_train_fin)
    
    finalist_model = xgb.train(params, dtrain_fin, num_boost_round=100)
    
    return model, finalist_model

model, finalist_model = load_models()

# --- Initialize session_state Defaults ---
if "class_25" not in st.session_state:
    st.session_state.class_25 = None
if "round" not in st.session_state:
    st.session_state.round = "Prelims"
if "week" not in st.session_state:
    st.session_state.week = None
if "captions" not in st.session_state:
    st.session_state.captions = "No"
if "competed" not in st.session_state:
    st.session_state.competed = "Yes"

# --- Callback Functions (Must Be Defined Before UI) ---
def update_class():
    st.session_state.class_25 = st.session_state.class_select

def update_round():
    st.session_state.round = st.session_state.round_select

def update_week():
    st.session_state.week = st.session_state.week_select

def update_captions():
    st.session_state.captions = st.session_state.captions_select

def update_previous():
    st.session_state.competed = st.session_state.competed_select

# --- UI Elements ---
classes = ['Independent A', 'Scholastic A', 'Independent Open', 'Scholastic Open', 'Independent World', 'Scholastic World']
weeks = ['Week 1: 2/8-9', 'Week 2: 2/15-16', 'Week 3: 2/22-23', 'Week 4: 3/1-2', 'Week 5: 3/8-9', 'Week 6: 3/15-16', 'Week 7: 3/22-23']

col1, col2, col3 = st.columns((1, 1, 1))
with col1:
    st.selectbox("2025 Class", classes, key="class_select", index=None, on_change=update_class)
    st.radio("Round", ["Prelims", "Finals"], key="round_select", on_change=update_round)

with col2:
    st.selectbox("Show Week", weeks, key="week_select", index=None, on_change=update_week)
    st.radio("Enter Caption Scores?", ["No", "Yes"], key="captions_select", on_change=update_captions)

# --- Caption Inputs ---
if st.session_state.captions == "Yes":
    with col1:
        ea_tot_sc = st.number_input("Equipment Analysis Total Score (out of 20 points)", min_value=0.0, max_value=20.0, format="%0.2f")
        da_tot_sc = st.number_input("Design Analysis Total Score (out of 20 points)", min_value=0.0, max_value=20.0, format="%0.2f")
    with col2:
        ma_tot_sc = st.number_input("Movement Analysis Total Score (out of 20 points)", min_value=0.0, max_value=20.0, format="%0.2f")
        tot_ge_sc = st.number_input("Total GE Score (out of 40 points)", min_value=0.0, max_value=40.0, format="%0.2f")
    subtot_sc = ea_tot_sc + ma_tot_sc + da_tot_sc + tot_ge_sc
    with col1:
        st.write('Total Subtotal score: ', "{:.3f}".format(subtot_sc))
    with col2:
        st.write('')
else:
    with col1:
        subtot_sc = st.number_input("Subtotal Score", min_value=0.0, max_value=100.0, format="%0.2f")
    ea_tot_sc = subtot_sc * 0.2
    ma_tot_sc = subtot_sc * 0.2
    da_tot_sc = subtot_sc * 0.2
    tot_ge_sc = subtot_sc * 0.4
    with col2:
        st.write('\n\n\n')

# --- Previous Year Inputs ---
with col2:
    st.write('')
with col1:
    prv_class = st.selectbox("Previous Class", classes, index=None)
with col2:
    st.radio("Did this guard compete last season?", ["Yes", "No"], key="competed_select", on_change=update_previous, horizontal=True)
    if st.session_state.competed == "Yes":
        with col1:
            prv_wc_round = st.radio("Previous WC Round", ["Prelims", "Semifinals", "Finals"], horizontal=True)
            prv_fin_score = st.number_input("Previous Final Score", min_value=0.0, max_value=100.0, format="%0.2f")
            prv_fin_place = st.number_input("Previous Final Placement", min_value=1, max_value=50, step=1)
            if prv_wc_round == "Prelims":
                prv_wc_round = 1
            if prv_wc_round == "Semifinals":
                prv_wc_round = 2
            if prv_wc_round == "Finals":
                prv_wc_round = 3
    else:
        prv_wc_round, prv_fin_score, prv_fin_place = 0, 0.0, 0

# --- Seed Score Calculation ---
week_offsets = {"1": 9.0, "2": 7.5, "3": 6.0, "4": 4.5, "5": 3.0, "6": 1.5, "7": 0.0}
week_num = st.session_state.week.split()[1].strip(":") if st.session_state.week else "7"
seed = subtot_sc + week_offsets.get(week_num, 0.0)

# --- Predict Button ---
if st.button("Predict Final Score"):
    input_data = pd.DataFrame([{
        'Round_Numb': 1 if st.session_state.round == "Prelims" else 3,
        'Class Numb': classes.index(st.session_state.class_25) + 1 if st.session_state.class_25 else 0,
        'EA_Tot_Sc': ea_tot_sc, 'MA_Tot_Sc': ma_tot_sc, 'DA_Tot_Sc': da_tot_sc,
        'Tot_GE_Sc': tot_ge_sc, 'Subtot_Sc': subtot_sc, 'Seed Score': seed,
        'Prv Class': classes.index(prv_class) + 1 if prv_class else 0,
        'Prv WC Round': prv_wc_round,
        'Prv Fin Score': prv_fin_score, 'Prv Fin Place': prv_fin_place
    }])

    dinput = xgb.DMatrix(input_data)
    prediction = model.predict(dinput)[0] if model else "Model not available"
    finalist_prob = finalist_model.predict(dinput)[0] if finalist_model else "Model not available"
    finalist_percentage = min(finalist_prob * 100, 100)

    st.subheader(f"Predicted Final Score: {prediction:.2f}")
    st.subheader(f"Odds of Making Finals: {finalist_percentage:.2f}%")
