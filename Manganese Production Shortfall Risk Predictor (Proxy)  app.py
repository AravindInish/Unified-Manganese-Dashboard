
import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import os

# Suppress warnings from shap or other libraries if they are too verbose
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Constants ---
MODEL_FILENAME = 'manganese_shortfall_model.pkl'
PREPROCESSOR_FILENAME = 'manganese_shortfall_preprocessor.pkl'
X_TRAIN_COLS_FILENAME = 'X_train_columns.pkl'

KAGGLE_DATASET_CSV_NAME = 'MiningProcess_Flotation_Plant_Database.csv'
# Correctly reference the path where the Kaggle dataset was downloaded
KAGGLE_DATASET_PATH = os.path.join('/kaggle/input/quality-prediction-in-a-mining-process', KAGGLE_DATASET_CSV_NAME)

target_iron_concentrate = 65.0 # From notebook
risk_thresholds = {
    'LOW': (0, 0.30),
    'MEDIUM': (0.30, 0.60),
    'HIGH': (0.60, 0.80),
    'CRITICAL': (0.80, 1.01) # 1.01 to include 100%
}

# --- Helper Functions (Copied from Notebook) ---
@st.cache_data # Cache data loading for performance
def load_and_preprocess_historical_data():
    """
    Loads and preprocesses the historical dataset to be used as context
    for feature engineering of new predictions.
    """
    try:
        df_hist = pd.read_csv(KAGGLE_DATASET_PATH)
    except FileNotFoundError:
        st.error(f"Historical dataset not found at {KAGGLE_DATASET_PATH}. Please ensure the file is present.")
        st.stop()

    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_hist = df_hist.set_index('date').sort_index()

    numeric_cols_to_convert = df_hist.select_dtypes(include='object').columns.tolist()
    for col in numeric_cols_to_convert:
        df_hist[col] = df_hist[col].str.replace(',', '.', regex=False).astype(float)

    df_hist.drop_duplicates(inplace=True)
    return df_hist

def create_features(dataframe):
    df_fe = dataframe.copy()

    df_fe['hour'] = df_fe.index.hour
    df_fe['dayofweek'] = df_fe.index.dayofweek
    df_fe['dayofmonth'] = df_fe.index.day
    df_fe['month'] = df_fe.index.month
    df_fe['quarter'] = df_fe.index.quarter
    df_fe['year'] = df_fe.index.year

    process_cols = [
        '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow',
        'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
        'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow',
        'Flotation Column 03 Air Flow', 'Flotation Column 04 Air Flow',
        'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow',
        'Flotation Column 07 Air Flow', 'Flotation Column 01 Level',
        'Flotation Column 02 Level', 'Flotation Column 03 Level',
        'Flotation Column 04 Level', 'Flotation Column 05 Level',
        'Flotation Column 06 Level', 'Flotation Column 07 Level',
        '% Iron Concentrate'
    ]

    for col in process_cols:
        df_fe[f'{col}_lag1'] = df_fe[col].shift(1)

    window_size = 3
    for col in process_cols:
        df_fe[f'{col}_rollmean{window_size}'] = df_fe[col].rolling(window=window_size).mean()
        df_fe[f'{col}_rollstd{window_size}'] = df_fe[col].rolling(window=window_size).std()

    df_fe.dropna(inplace=True)
    return df_fe

def assign_risk_level(probability, thresholds):
    for level, (lower, upper) in thresholds.items():
        if lower <= probability < upper:
            return level
    return "UNKNOWN"

def plot_risk_gauge(probability, risk_level_str, target_production, expected_production, expected_shortfall, title="Proxy Shortfall Risk"): # Changed title for app
    if probability is None or not (0 <= probability <= 1):
        # st.warning("Invalid probability for gauge plot.") # Using st.warning for Streamlit
        return None # Return None if invalid

    prob_percent = probability * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "red"}}, # Example reference
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'green', 'name': 'LOW'},
                {'range': [30, 60], 'color': 'yellow', 'name': 'MEDIUM'},
                {'range': [60, 80], 'color': 'orange', 'name': 'HIGH'},
                {'range': [80, 100], 'color': "red", 'name': 'CRITICAL'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prob_percent}}))

    fig.update_layout(
        paper_bgcolor="lavender",
        font={'color': "darkblue", 'family': "Arial"},
        annotations=[
            dict(text=f"Risk Level: {risk_level_str}", x=0.5, y=0.1, font_size=16, showarrow=False),
            dict(text=f"Target Prod: {target_production}", x=0.5, y=0.2, font_size=12, showarrow=False),
            dict(text=f"Expected Prod: {expected_production}", x=0.5, y=0.15, font_size=12, showarrow=False),
            dict(text=f"Expected Shortfall: {expected_shortfall}", x=0.5, y=0.05, font_size=12, showarrow=False)
        ]
    )
    return fig # Return figure instead of showing


def predict_shortfall(input_data_df: pd.DataFrame, model, scaler, x_train_cols, risk_thresholds, original_df_for_features):
    """
    Predicts shortfall probability and risk level for new input data.

    Args:
        input_data_df (pd.DataFrame): A DataFrame containing new input data, similar structure to original data (e.g., one row).
                                      Must include a 'date' column or be indexed by date.
        model: The trained machine learning model.
        scaler: The trained scaler for feature scaling.
        x_train_cols (pd.Index): Column names of features used during model training.
        risk_thresholds (dict): Dictionary defining risk level thresholds.
        original_df_for_features (pd.DataFrame): The original, preprocessed DataFrame used for training,
                                                 needed to compute lagged/rolling features for new data.
                                                 Must be indexed by date, and contain '% Iron Concentrate'.

    Returns:
        dict: A dictionary containing prediction results.
    """
    if 'date' in input_data_df.columns:
        input_data_df['date'] = pd.to_datetime(input_data_df['date'])
        input_data_df = input_data_df.set_index('date').sort_index()
    else:
        # Assume input_data_df is already date-indexed or has a placeholder date
        if not isinstance(input_data_df.index, pd.DatetimeIndex):
            st.error("Input data must have a DatetimeIndex.")
            return {}

    # Combine new data with historical data to compute lagged/rolling features
    combined_df = pd.concat([
        original_df_for_features.drop(columns=['proxy_shortfall', 'proxy_shortfall_flag'], errors='ignore'),
        input_data_df
    ])

    engineered_combined_df = create_features(combined_df)

    # Extract features for the new input row only. It should be the last row.
    new_data_index = input_data_df.index

    # Filter to only X_train_cols, ensuring consistency
    X_new_features = engineered_combined_df.loc[new_data_index][x_train_cols] # Use x_train_cols here

    # Align columns with training data and scale
    X_new_processed = scaler.transform(X_new_features) # Use X_new_features here

    # Predict probability
    probability = model.predict_proba(X_new_processed)[:, 1][0]

    # Assign risk level
    risk_level = assign_risk_level(probability, risk_thresholds)

    # SHAP for major risk factors
    explainer_single = shap.TreeExplainer(model)
    shap_values_single = explainer_single.shap_values(X_new_features) # Use X_new_features for SHAP

    if isinstance(shap_values_single, list):
        shap_df = pd.DataFrame({'feature': x_train_cols, 'shap_value': shap_values_single[1][0]})
    else:
        shap_df = pd.DataFrame({'feature': x_train_cols, 'shap_value': shap_values_single[0] if shap_values_single.ndim > 1 else shap_values_single})

    shap_df['abs_shap_value'] = shap_df['shap_value'].abs()
    major_contributors_local = shap_df.sort_values(by='abs_shap_value', ascending=False).head(5)
    
    # Return a list of dictionaries, each with 'feature' and 'impact' (shap_value)
    major_risk_factors_output = [{
        "feature": row['feature'], 
        "impact": row['shap_value']
    } for index, row in major_contributors_local.iterrows()]

    return {
        "probability": f"{probability:.2f}",
        "risk_level": risk_level,
        "expected_production": "N/A (classification model)", # Placeholder
        "target_production": target_iron_concentrate,
        "expected_shortfall": "N/A (classification model)", # Placeholder
        "major_risk_factors": major_risk_factors_output
    }

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Shortfall Risk Predictor ↗️", page_icon="⚠️")

st.title("Manganese Production Shortfall Risk Predictor (Proxy) ↗️⚠️")
st.markdown("This application predicts the risk of a **proxy shortfall** in Iron Concentrate quality based on various process parameters. "
            "_Disclaimer: This model uses a proxy target (% Iron Concentrate) due to data limitations, not actual Manganese production._")

# --- Load Assets ---
@st.cache_resource # Cache the model and scaler loading
def load_model_assets():
    try:
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(PREPROCESSOR_FILENAME)
        x_train_cols = joblib.load(X_TRAIN_COLS_FILENAME)
        return model, scaler, x_train_cols
    except FileNotFoundError as e:
        st.error(f"Error loading model assets: {e}. Please ensure '{MODEL_FILENAME}', '{PREPROCESSOR_FILENAME}', and '{X_TRAIN_COLS_FILENAME}' are in the same directory as this app.")
        st.stop()

model, scaler, x_train_cols = load_model_assets()
historical_df = load_and_preprocess_historical_data()

# --- Sidebar for Input ---
st.sidebar.header("Input Process Parameters 👇")

# Define the original raw feature columns for input
# These are the columns expected by the `create_features` function before any FE
raw_input_columns = [
    '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow',
    'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
    'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow',
    'Flotation Column 03 Air Flow', 'Flotation Column 04 Air Flow',
    'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow',
    'Flotation Column 07 Air Flow', 'Flotation Column 01 Level',
    'Flotation Column 02 Level', 'Flotation Column 03 Level',
    'Flotation Column 04 Level', 'Flotation Column 05 Level',
    'Flotation Column 06 Level', 'Flotation Column 07 Level',
    '% Iron Concentrate', # This is a current process output, but also an input feature
    '% Silica Concentrate'
]

input_data = {}
for col in raw_input_columns:
    # Use st.sidebar for inputs
    default_value = historical_df[col].mean() # Use mean of historical data as default
    input_data[col] = st.sidebar.number_input(f"**{col}**", value=float(default_value), format="%.4f", key=col)

# Add a button to trigger prediction
predict_button = st.sidebar.button("Predict Shortfall Risk ✨")

# --- Main Content Area for Results ---
st.header("Prediction Results 📊")

if predict_button:
    if historical_df is None or model is None or scaler is None or x_train_cols is None:
        st.error("Model assets or historical data not loaded. Cannot make predictions.")
    else:
        # Create a DataFrame from the input data, mimicking a single new timestamp
        # Use the last date from historical data + 1 hour as the new timestamp
        last_historical_date = historical_df.index.max()
        new_timestamp = last_historical_date + pd.Timedelta(hours=1)

        input_df_for_prediction = pd.DataFrame([input_data], index=[new_timestamp])
        input_df_for_prediction.index.name = 'date'

        with st.spinner("Calculating prediction and risk factors..."):
            prediction_output = predict_shortfall(
                input_data_df=input_df_for_prediction,
                model=model,
                scaler=scaler,
                x_train_cols=x_train_cols, # Pass X_train columns explicitly
                risk_thresholds=risk_thresholds,
                original_df_for_features=historical_df
            )

        if prediction_output:
            # Convert probability to float for comparison and display
            predicted_probability_float = float(prediction_output['probability'])

            st.subheader(f"Predicted Probability: {predicted_probability_float:.2f} 📈")
            st.subheader(f"Risk Level: {prediction_output['risk_level']} {'🟢' if prediction_output['risk_level'] == 'LOW' else ('🟡' if prediction_output['risk_level'] == 'MEDIUM' else ('🟠' if prediction_output['risk_level'] == 'HIGH' else '🔴'))}")

            st.write(f"**Target % Iron Concentrate:** {prediction_output['target_production']}% ")
            st.write(f"**Expected Production:** {prediction_output['expected_production']}")
            st.write(f"**Expected Shortfall:** {prediction_output['expected_shortfall']}")

            st.markdown("### Major Risk Factors 💡")
            if prediction_output['major_risk_factors']:
                for factor_info in prediction_output['major_risk_factors']:
                    st.write(f"*   **{factor_info['feature']}**: Impact {factor_info['impact']:.4f}")
            else:
                st.write("No major risk factors identified for this prediction.")

            st.markdown("### Risk Gauge 📊")
            risk_gauge_fig = plot_risk_gauge(
                probability=predicted_probability_float,
                risk_level_str=prediction_output['risk_level'],
                target_production=prediction_output['target_production'],
                expected_production=prediction_output['expected_production'],
                expected_shortfall=prediction_output['expected_shortfall']
            )
            if risk_gauge_fig:
                st.plotly_chart(risk_gauge_fig, use_container_width=True)
            else:
                st.error("Could not generate risk gauge.")
        else:
            st.error("Prediction could not be generated. Please check input values and server logs.")

else:
    st.info("Enter the parameters in the sidebar and click 'Predict Shortfall Risk' to see the results!")

st.markdown("--- ")
st.markdown("Developed for SIH26009 - Manganese Production Shortfall Risk Prediction Model (Proxy)")
