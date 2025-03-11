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

# --- Initialize session_state Defaults ---
if "class_25" not in st.session_state:
    st.session_state.class_25 = None
if "round" not in st.session_state:
    st.session_state.round = "Prelims"
if "week" not in st.session_state:
    st.session_state.week = None
if "captions" not in st.session_state:
    st.session_state.captions = "No"

# --- Callback Functions (Defined Before UI) ---
def update_class():
    st.session_state.class_25 = st.session_state.class_select

def update_round():
    st.session_state.round = st.session_state.round_select

def update_week():
    st.session_state.week = st.session_state.week_select

def update_captions():
    st.session_state.captions = st.session_state.captions_select

# --- UI Elements ---
classes = ['Independent A', 'Scholastic A', 'Independent Open', 'Scholastic Open', 'Independent World', 'Scholastic World']
weeks = ['Week 1: 2/8-9', 'Week 2: 2/15-16', 'Week 3: 2/22-23', 'Week 4: 3/1-2', 'Week 5: 3/8-9', 'Week 6: 3/15-16', 'Week 7: 3/22-23']

col1, col2, col3 = st.columns((1, 1, 1))
with col1:
    st.selectbox("2025 Class", classes, key="class_select", index=None, on_change=update_class)
    st.radio("Round", ["Prelims", "Finals"], key="round_select", on_change=update_round)

with col2:
    st.selectbox("Show Week", weeks, key="week_select", index=None, on_change=update_week)
    st.radio("Enter Caption Scores?", ["Yes", "No"], key="captions_select", on_change=update_captions)

# --- Caption Inputs ---
if st.session_state.captions == "Yes":
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
    }])

    dinput = xgb.DMatrix(input_data)
    prediction = model.predict(dinput)[0] if model else "Model not available"
    finalist_prob = finalist_model.predict(dinput)[0] if finalist_model else "Model not available"
    finalist_percentage = min(finalist_prob * 100, 100)

    st.subheader(f"Predicted Final Score: {prediction:.3f}")
    st.subheader(f"Odds of Making Finals: {finalist_percentage:.2f}%")
