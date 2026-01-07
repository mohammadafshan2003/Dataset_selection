import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, precision_recall_curve
import joblib
import streamlit as st
import datetime
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("transactions.csv")

df['transaction_time'] = pd.to_datetime(df['transaction_time'])

df['hour'] = df['transaction_time'].dt.hour
df['day_of_week'] = df['transaction_time'].dt.dayofweek   # 0=Monday, 6=Sunday
df['month'] = df['transaction_time'].dt.month

# A. Hour (Cycle = 24)
hour=st.number_input(label="enter hour",min_value=0,max_value=23)
day=st.selectbox("select day of transaction",options=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
month=st.number_input(label="month of transaction",min_value=1,max_value=12)
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
day_num = day_map[day]


hour_sin=np.sin(2 * np.pi * hour / 24)
hour_cos=np.cos(2 * np.pi * hour / 24)

# B. Day of Week (Cycle = 7)
day_sin=np.sin(2 * np.pi * day_num / 7)
day_cos=np.cos(2 * np.pi *  day_num / 7)

# C. Month (Cycle = 12)
month_sin=np.sin(2 * np.pi * month / 12)
month_cos= np.cos(2 * np.pi * month / 12)

df = df.drop(columns=["hour", "day_of_week", "month","transaction_time","transaction_id","user_id"])

account_age_days = st.number_input(
    "Account Age (days)", min_value=0, step=1
)

total_transactions_user = st.number_input(
    "Total Transactions by User", min_value=0, step=1
)

avg_amount_user = st.number_input(
    "Average Transaction Amount (User)", min_value=0.0
)

amount = st.number_input(
    "Current Transaction Amount", min_value=0.0
)

shipping_distance_km = st.number_input(
    "Shipping Distance (km)", min_value=0.0
)

# ================= Categorical Inputs =================
country = st.selectbox(
    "User Country",
    ["US", "GB", "FR", "NL", "TR", "PL", "RO", "DE", "ES", "IT"]
)

bin_country = st.selectbox(
    "BIN Country",
    ["US", "GB", "FR", "NL", "TR", "PL", "RO", "DE", "ES", "IT"]
)

channel = st.selectbox(
    "Transaction Channel",
    ["web", "app"]
)

merchant_category = st.selectbox(
    "Merchant Category",
    ["electronics", "travel", "grocery", "gaming", "fashion"]
)
promo_used = st.selectbox("Promo Used?", [0, 1])
avs_match = st.selectbox("AVS Match?", [0, 1])
cvv_result = st.selectbox("CVV Result Match?", [0, 1])
three_ds_flag = st.selectbox("3DS Authentication Used?", [0, 1])
country_mismatch = st.selectbox("Country Mismatch?", [0, 1])
df['country_mismatch'] = df['country'] != df['bin_country']
df['country_mismatch'] = df['country_mismatch'].astype(int)


#splitting the data
X_train = df.drop("is_fraud", axis = 1)
inputss = pd.DataFrame({
    "day_sin":[day_sin],
    "day_cos":[day_cos],
    "month_sin":[month_sin],
    "month_cos":[month_cos],
    "hour_sin":[hour_sin],
    "hour_cos":[hour_cos],
})
X_trains=pd.concat([inputss,X_train],axis=1)
y = df["is_fraud"]


encoding = [
    "country","bin_country","channel","merchant_category"
]
input_data = pd.DataFrame({
    "country": [country],
    "bin_country": [bin_country],
    "channel": [channel],
    "merchant_category": [merchant_category]
})
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), encoding)
    ],
    remainder="passthrough")

#Transform X_train
X_train_transformed=preprocessor.fit_transform(X_trains)
# Convert to DataFrame
X_train_df = pd.DataFrame(
    X_train_transformed,
    columns=preprocessor.get_feature_names_out(),
)


input = pd.DataFrame({
    "day_sin":[day_sin],
    "day_cos":[day_cos],
    "month_sin":[month_sin],
    "month_cos":[month_cos],
    "hour_sin":[hour_sin],
    "hour_cos":[hour_cos],
    "account_age_days": [account_age_days],
    "total_transactions_user": [total_transactions_user],
    "avg_amount_user": [avg_amount_user],
    "amount": [amount],
    "promo_used": [promo_used],
    "avs_match": [avs_match],
    "cvv_result": [cvv_result],
    "three_ds_flag": [three_ds_flag],
    "shipping_distance_km": [shipping_distance_km],
    "country_mismatch": [country_mismatch]
})
dataframe=pd.concat([input_data,input],axis=1)
X_test_transformed=preprocessor.transform(dataframe)
X_test_df = pd.DataFrame(
    X_test_transformed,
    columns=preprocessor.get_feature_names_out(),
    #index=X_test_transformed.index
)
but=st.button("submit")
if but:
    modelss=joblib.load("xgb_fraud_model_with_threshold.pkl")
    trained_order =modelss["model"].feature_names_in_
    X_test = X_test_df.reindex(columns=trained_order)
    st.write(X_test)
    pred=modelss["model"].predict(X_test)
    if pred==0:
        st.write(f"Legitimate and value is {pred}")
    else:
        st.write(f"Fraud and value is {pred}")