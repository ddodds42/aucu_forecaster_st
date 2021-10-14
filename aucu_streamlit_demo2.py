import category_encoders as ce
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict, \
    RandomizedSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
import streamlit as st
import sys
from PIL import Image


# 1 - Train - Test Split
oth_features = ['ApplicationType', 'BusinessType', 'PostalCode']
crtd_features  = [
  'Created_DayOfYear', 'Created_Month', 'Created_DayOfMonth',
  'Created_DayOfWeek'
]
features = oth_features + crtd_features

target = "TotalDaysCreationToIssuance_Ints"

test0 = pd.read_csv("./Datasets/8_test_aucu_since2017_encoded.csv")
trainval0 = pd.read_csv("./Datasets/8_trainval_aucu_since2017_encoded.csv")

X_trainval0 = trainval0[features]
y_trainval0 = trainval0[target]
X_test0 = test0[features]
y_test0 = test0[target]


# 1.5 - Target Encoding
# df = pd.read_csv("./Datasets/8certificateOfUse_IssuedSince2017_YearCol.csv")
df = pd.read_csv("./Datasets/8_anon_certificateOfUse_IssuedSince2017.csv")

date_features = [
    'Created','Submitted','InReview','Billed','Paid','UnderInspection','Ready',
    'Issued','Printed'
]
duration_features = [
    'DaysInCreatedStatus','DaysInReview','DaysInBilled','DaysInPaid',
    'DaysInUnderInspection','DaysInReady','DaysInIssued',
    'TotalDaysCreationToPrint_Original'
]
postal_features = ["PostalCode", "PostalCodeStratify"]

def custom_dtypes(
    df, to_dtime = date_features, to_int = duration_features,
    to_str = postal_features
    ):
  
  ## function to correct certain column's datatypes.
  
  df[to_dtime] = df[to_dtime].apply(
    pd.to_datetime, errors='coerce'
  )

  df[to_int] = df[to_int].astype(pd.Int64Dtype())

  df[to_str] = df[to_str].astype(str)

  return df.dtypes

custom_dtypes(df)

oth_features = ['ApplicationType', 'BusinessType', 'PostalCode']
crtd_features  = [
  'Created_DayOfYear', 'Created_Month', 'Created_DayOfMonth',
  'Created_DayOfWeek'
]
features = oth_features + crtd_features

target = "TotalDaysCreationToIssuance_Ints"

df_enc = df[oth_features]

encoder = ce.TargetEncoder(min_samples_leaf=1, smoothing=1)
encoder.fit(df_enc, df[target])


# 2 - Random Forest 9
best_rf = RandomForestRegressor(
  bootstrap=True, ccp_alpha=1.6568212939471354, criterion='mse', max_depth=10,
  max_features=0.9937868102260314, max_leaf_nodes=None, max_samples=None, 
  min_impurity_decrease=1.2839842281707012, min_impurity_split=None,
  min_samples_leaf=135, min_samples_split=2, min_weight_fraction_leaf=0.0,
  n_estimators=70, n_jobs=-1, oob_score=False, random_state=42, verbose=0,
  warm_start=False
)

best_rf.fit(X_trainval0, y_trainval0)


# 3 - Demo Functions
def overestimator(pred):
  ## rounds wait-time estimate to nearest 5, returns a 15 day estimate range.
  add = 5 - (pred % 5)
  tell = pred + add
  long = tell + 15
  return [tell, long]

def wait_time(atype, btype, zipcode):

  ## uses random forest estimator to return a predicted AU/CU wait time.

  ts0 = pd.Timestamp.today()

  trial_input = {
    "ApplicationType":atype, "BusinessType":btype, "PostalCode":zipcode
  }

  X_frame = pd.DataFrame(trial_input, index=[0])
  X_enc = encoder.transform(X_frame)

  X_enc["Created_DayOfYear"] = ts0.dayofyear
  X_enc["Created_Month"] = ts0.month
  X_enc["Created_DayOfMonth"] = ts0.day
  X_enc["Created_DayOfWeek"] = ts0.dayofweek

  demodays = best_rf.predict(X_enc)
  ready = overestimator(demodays)

  answer = (
      "Your " + atype + " will be ready in approximately " +
      "%.0f" % ready[0] + "-" + "%.0f" % ready[1] + " days."
      )

  return answer


# 4 - Streamlit Demo Code
st_subh = (
    "Use this predictor to forecast how long it might tike to get a business " +
    "license from the City of Miami."
    )

st.title("Thinking about starting a business in Miami?")
st.subheader(st_subh)

app_type = st.selectbox(
    "Application Type", options = list(np.sort(df["ApplicationType"].unique()))
    )
biz_type = st.selectbox(
    "Business Type", options = list(np.sort(df["BusinessType"].unique()))
    )
zip_code = st.selectbox(
    "Zip Code", options = list(np.sort(df["PostalCode"].unique()))
    )

forecast = wait_time(app_type, biz_type, zip_code)

st.subheader(forecast)
st.subheader("\n")

img = Image.open("seal-of-miami-logo.png")
st.image(img, use_column_width="auto")