
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Load Model and Preprocessor ---
# Ensure these files exist in the same directory as the app.py or provide full paths.
model_filename = 'best_manganese_prospectivity_model.joblib'
preprocessor_filename = 'manganese_prospectivity_preprocessor.joblib'

if not os.path.exists(model_filename) or not os.path.exists(preprocessor_filename):
    st.error("Model or preprocessor files not found! Please ensure 'best_manganese_prospectivity_model.joblib' and 'manganese_prospectivity_preprocessor.joblib' are in the same directory as app.py")
    st.stop()

loaded_preprocessor = joblib.load(preprocessor_filename)
loaded_model = joblib.load(model_filename)

# --- Define the prediction function ---
# This is a slightly modified version of the function from the notebook
# to ensure it uses the loaded objects and is self-contained for the app.
def predict_manganese_prospectivity_app(data_point: dict, preprocessor_obj, model_obj, numerical_features, categorical_features):
    if not isinstance(data_point, list):
        data_point = [data_point]
    input_df = pd.DataFrame(data_point)

    expected_columns = numerical_features + categorical_features

    for col in categorical_features:
        if col not in input_df.columns:
            input_df[col] = 'Unknown'

    for col in numerical_features:
        if col not in input_df.columns:
            input_df[col] = 0.0 
            
    input_df = input_df[expected_columns]

    processed_input = preprocessor_obj.transform(input_df)
    probabilities = model_obj.predict_proba(processed_input)[:, 1]
    return probabilities

# --- Get unique categories for dropdowns ---
# These should ideally be derived from the training data, but for this app,
# we'll use a hardcoded set or dynamically extract from a sample df_processed.
# For a robust app, store these as a separate artifact from training.

# NOTE: For a real deployment, these unique values should be saved along with the preprocessor
# and loaded here, rather than relying on global `df_processed`.
# For demonstration, we'll assume `df_processed` and `categorical_features` are available from the notebook's execution context.

# Assuming `df_processed` and `categorical_features` are available in the Colab environment
# and that the app.py is run after the notebook cells have defined them.
# In a production setting, you'd save these unique values to a file (e.g., JSON) during training.

# Example of how to get unique values if df_processed is NOT available here (uncomment and replace with actual values)
# unique_countries = ['United States', 'India', 'Canada', 'Mexico', 'Brazil', 'Unknown'] 
# unique_states = ['California', 'Odisha', 'Ontario', 'Chihuahua', 'Minas Gerais', 'Unknown']
# ... and so on

# For this demonstration within Colab, we'll try to use the global variables if they exist
# or use some reasonable defaults if this script is run standalone without the full notebook context.

try:
    # Accessing global variables from the Colab notebook's execution environment
    # This part depends on how this `app.py` is executed within Colab. 
    # If run via `!streamlit run app.py`, it's a separate process and these globals won't be available.
    # Thus, for robust app, these should be loaded from saved files.
    _numerical_features = ['latitude', 'longitude'] # These are constant
    _categorical_features = ['country', 'state', 'com_type', 'oper_type', 'prod_size', 'dev_stat'] # These are constant

    # In a production app, save these unique values as a JSON or pickle during training
    # and load them here.
    # For this example, if running directly in Colab after the notebook, these might be accessible.
    # However, if 'app.py' is truly run as a separate script, these will cause an error.
    # Let's create a dummy DataFrame to ensure it runs even if `df_processed` is not in scope
    
    # Fallback/dummy values for dropdowns if full df_processed isn't accessible directly
    # It's highly recommended to store these unique values from your training set separately.
    unique_countries = ['Unknown', 'United States', 'Canada', 'Mexico', 'Peru', 'Brazil', 'Chile', 'Argentina', 'Bolivia', 'Russia', 'Australia', 'China', 'India', 'South Africa', 'Venezuela', 'Ecuador', 'Guyana', 'France', 'Costa Rica', 'Cuba'] # Top 20 from EDA
    unique_states = ['Unknown', 'California', 'Colorado', 'Oregon', 'Nevada', 'Arizona', 'Utah', 'Idaho', 'Washington', 'Montana', 'New Mexico', 'Wyoming', 'Alaska', 'Missouri', 'Wisconsin', 'Georgia', 'Pennsylvania', 'North Carolina', 'South Carolina', 'Virginia'] # Top 20 from EDA
    unique_com_types = ['Unknown', 'M', 'N', 'Unknown', 'C', 'L'] # from EDA
    unique_oper_types = ['Unknown', 'Surface', 'Underground', 'Surface-Underground', 'Placer', 'Unknown', 'Open Pit', 'Strip Mine', 'Dredge', 'Heap Leach', 'In-Situ Leach'] # from EDA
    unique_prod_sizes = ['Unknown', 'L', 'M', 'S', 'Unknown', 'P', 'D'] # from EDA
    unique_dev_stats = ['Unknown', 'Producer', 'Prospect', 'Occurrence', 'Past Producer', 'Inactive', 'Developed Prospect'] # from EDA

    # Dynamically generate if df_processed is present in scope (e.g. if the cell is run after all previous cells)
    try:
        global df_processed # Try to access the global df_processed from the notebook
        if 'df_processed' in globals():
            unique_countries = sorted(df_processed['country'].unique().tolist())
            unique_states = sorted(df_processed['state'].unique().tolist())
            unique_com_types = sorted(df_processed['com_type'].unique().tolist())
            unique_oper_types = sorted(df_processed['oper_type'].unique().tolist())
            unique_prod_sizes = sorted(df_processed['prod_size'].unique().tolist())
            unique_dev_stats = sorted(df_processed['dev_stat'].unique().tolist())
            st.success("Using unique categories from the notebook's DataFrame. ✨")
    except NameError:
        st.warning("Could not access `df_processed` for dynamic unique categories. Using default lists. For full functionality, run notebook cells defining `df_processed` first, or provide pre-saved unique lists.")


except Exception as e:
    st.error(f"Error setting up categories: {e}")
    st.stop()


# Streamlit App Layout
st.set_page_config(layout="wide", page_title="Manganese Prospectivity Predictor 🗺️⛏️")

st.title('Manganese Prospectivity Prediction Model 🌍✨')
st.markdown("Welcome to the Manganese Prospectivity Predictor! Use the sidebar to input geological and geographical data to estimate the probability of manganese mineralization. 🧐")

# --- Sidebar for User Input --- 
st.sidebar.header('Input Features for Prediction 👇')

with st.sidebar:
    st.markdown("### Location Details 📍")
    latitude = st.slider('Latitude', min_value=-90.0, max_value=90.0, value=34.05, step=0.01)
    longitude = st.slider('Longitude', min_value=-180.0, max_value=180.0, value=-118.24, step=0.01)
    
    st.markdown("### Geological & Operational Data ⛏️")
    country = st.selectbox('Country', options=unique_countries, index=unique_countries.index('United States') if 'United States' in unique_countries else 0)
    state = st.selectbox('State', options=unique_states, index=unique_states.index('California') if 'California' in unique_states else 0)
    com_type = st.selectbox('Commodity Type', options=unique_com_types, index=unique_com_types.index('M') if 'M' in unique_com_types else 0)
    oper_type = st.selectbox('Operation Type', options=unique_oper_types, index=unique_oper_types.index('Surface') if 'Surface' in unique_oper_types else 0)
    prod_size = st.selectbox('Production Size', options=unique_prod_sizes, index=unique_prod_sizes.index('M') if 'M' in unique_prod_sizes else 0)
    dev_stat = st.selectbox('Development Status', options=unique_dev_stats, index=unique_dev_stats.index('Producer') if 'Producer' in unique_dev_stats else 0)

    st.markdown("### Ready to predict? 🚀")
    predict_button = st.button('Predict Prospectivity!')

# --- Main Content Area for Results --- 
st.header('Prediction Results 📊')

if predict_button:
    user_input = {
        'latitude': latitude,
        'longitude': longitude,
        'country': country,
        'state': state,
        'com_type': com_type,
        'oper_type': oper_type,
        'prod_size': prod_size,
        'dev_stat': dev_stat
    }

    # Ensure consistent numerical and categorical feature lists for the prediction function
    # These must match those used during preprocessor training.
    numerical_features_for_pred = ['latitude', 'longitude']
    categorical_features_for_pred = ['country', 'state', 'com_type', 'oper_type', 'prod_size', 'dev_stat']

    prediction_probability = predict_manganese_prospectivity_app(
        user_input, loaded_preprocessor, loaded_model, 
        numerical_features_for_pred, categorical_features_for_pred
    )[0]

    st.subheader(f"Prediction Probability: {prediction_probability:.4f} 📈")

    if prediction_probability > 0.5:
        st.success("Prospectivity: HIGH! 🎉 This area shows a high potential for Manganese mineralization.")
        st.balloons()
    else:
        st.info("Prospectivity: LOW. 📉 This area shows lower potential for Manganese mineralization.")

    st.write("--- ")
    st.write("Disclaimer: This model provides a probability estimate and should be used as a guiding tool. Further geological investigation is always recommended! 🧐")

else:
    st.info("Adjust the parameters in the sidebar and click 'Predict Prospectivity!' to see the results. 👆")
