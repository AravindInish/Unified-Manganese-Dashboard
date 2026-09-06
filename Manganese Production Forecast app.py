import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('xgb_manganese_model.joblib')

st.set_page_config(layout="wide")

st.title("🌍 Manganese Production Forecaster 📊")
st.markdown("### Predict Future Manganese Production for Countries Worldwide! ✨")

# Get the feature names from X_train during model training
# This assumes X_train.columns was available in the global scope when this cell was run
# In a real deployed app, you might save these feature names alongside the model or derive them.

# For this demonstration, we'll recreate a simplified version of X_train.columns
# based on the known features and country encoding.

# Manually define features for the input panel (should match X_train.columns)
# Numerical features
numerical_features = [
    'production_lag_1', 'production_lag_2', 'production_lag_3',
    'rolling_mean_3', 'rolling_std_3', 'growth_rate', 'volatility', 'year_int'
]

# Country features (all unique countries that were one-hot encoded, except the one dropped by drop_first=True)
# For simplicity, let's derive unique countries from df_manganese_fe if possible.
# If not, you'd need to explicitly list them or load from a saved list.
# Let's assume `df_manganese_fe` was available and we can get unique countries.
# In a deployed app, you'd load this list.

# Placeholder for unique countries (replace with actual logic if df_manganese_fe is available at runtime)
# For the purpose of writing the app script, we'll hardcode based on the notebook's X_train.columns

# From X_train.columns output in the previous cell:
all_feature_columns = ['production_lag_1', 'production_lag_2', 'production_lag_3', 'rolling_mean_3', 'rolling_std_3', 'growth_rate', 'volatility', 'year_int', 'country_Australia', 'country_Bolivia', 'country_Brazil', 'country_China', 'country_Colombia', 'country_Congo, D.R.', "country_Cote d'Ivoire", 'country_Egypt', 'country_Gabon', 'country_Georgia', 'country_Ghana', 'country_Guyana', 'country_India', 'country_Indonesia', 'country_Iran', 'country_Kazakhstan', 'country_Kenya', 'country_Malaysia', 'country_Mexico', 'country_Morocco', 'country_Myanmar', 'country_Namibia', 'country_Nigeria', 'country_Oman', 'country_Pakistan', 'country_Peru', 'country_Romania', 'country_Russia', 'country_Russia, Europe', 'country_Senegal', 'country_South Africa', 'country_Sudan', 'country_Thailand', 'country_Türkiye', 'country_Ukraine', 'country_Vietnam', 'country_Zambia']

# Extract country names from feature columns
country_options = sorted([col.replace('country_', '') for col in all_feature_columns if col.startswith('country_')])
country_options.insert(0, 'Angola') # Add the base country if 'drop_first=True' was used


# Sidebar for user inputs
st.sidebar.header("Input Features ⚙️")

selected_country = st.sidebar.selectbox("Select Country 🗺️", country_options)

production_lag_1 = st.sidebar.number_input("Production Lag 1 (Previous Year's Production) 📉", min_value=0.0, value=100000.0, step=1000.0)
production_lag_2 = st.sidebar.number_input("Production Lag 2 (2 Years Ago Production) 🗓️", min_value=0.0, value=90000.0, step=1000.0)
production_lag_3 = st.sidebar.number_input("Production Lag 3 (3 Years Ago Production) 🕰️", min_value=0.0, value=80000.0, step=1000.0)
rolling_mean_3 = st.sidebar.number_input("Rolling Mean (3-year) 📈", min_value=0.0, value=90000.0, step=1000.0)
rolling_std_3 = st.sidebar.number_input("Rolling Std Dev (3-year) 〰️", min_value=0.0, value=10000.0, step=100.0)
growth_rate = st.sidebar.number_input("Growth Rate (%) 🚀", value=5.0, step=0.1)
volatility = st.sidebar.number_input("Volatility (Std Dev) 🌪️", min_value=0.0, value=5000.0, step=100.0)
year_int = st.sidebar.number_input("Target Year 📅", min_value=2023, max_value=2050, value=2023, step=1)


# Create a dictionary for the input data
input_data = {
    'production_lag_1': production_lag_1,
    'production_lag_2': production_lag_2,
    'production_lag_3': production_lag_3,
    'rolling_mean_3': rolling_mean_3,
    'rolling_std_3': rolling_std_3,
    'growth_rate': growth_rate,
    'volatility': volatility,
    'year_int': year_int
}

# Add one-hot encoded country features
for country_col in country_options:
    # X_train.columns uses 'country_CountryName' format
    feature_name = f'country_{country_col}'
    if feature_name in all_feature_columns: # Ensure the feature exists in the model's expected columns
        input_data[feature_name] = 1 if country_col == selected_country else 0

# Ensure all features expected by the model are present, fill missing with 0
# This is crucial if `all_feature_columns` contains features not manually added above
for col in all_feature_columns:
    if col not in input_data:
        input_data[col] = 0 # Default to 0 for any missing feature


# Create a DataFrame from the input data, ensuring column order matches X_train
input_df = pd.DataFrame([input_data])
input_df = input_df[all_feature_columns] # Reorder columns to match model's training data

# Prediction button
if st.sidebar.button("Forecast Production! 🔮"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_formatted = f"{prediction:,.2f}"
        st.success(f"The forecasted manganese production for **{selected_country}** in **{year_int}** is: ")
        st.balloons()
        st.markdown(f"## ➡️ {prediction_formatted} tons! 🌟")
        st.info("*(Note: Negative predictions are possible with Linear Regression if data is sparse, but XGBoost often handles this better. Consider post-processing for real-world application.)* 💡")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e} 💔")
        st.warning("Please ensure all input values are reasonable. 🙏")

st.markdown("---")
st.markdown("Developed by ARAVIND INISH 🤖")
