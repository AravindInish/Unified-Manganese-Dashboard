
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import shap
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration and Constants ---
MODEL_DIR = "/content/manganese_models/"

# --- Model and Preprocessor Filenames ---
MODEL_FILENAMES = {
    "prospectivity_model": "best_manganese_prospectivity_model.joblib",
    "prospectivity_preprocessor": "manganese_prospectivity_preprocessor.joblib",
    "reserve_grade_pipeline": "manganese_reserve_model_grade.pkl",
    "reserve_viability_pipeline": "manganese_reserve_model_viability.pkl",
    "reserve_preprocessor_viability": "manganese_reserve_preprocessor_viability.pkl",
    "production_forecast_model": "maganese production forecost xgb_manganese_model.joblib", # Corrected name
    "shortfall_model": "manganese_shortfall_model.pkl",
    "shortfall_preprocessor": "manganese_shortfall_preprocessor.pkl",
    "shortfall_X_train_cols": "X_train_columns.pkl",
}

# --- Cached Functions for Loading Models and Preprocessors ---
@st.cache_resource
def load_model(file_path):
    """Loads a single joblib/pkl model or preprocessor with caching."""
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {file_path}. Please ensure all model artifacts are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model from {file_path}: {e}")
        st.stop()

@st.cache_resource
def load_all_models():
    """Loads all required models and preprocessors."""
    loaded_assets = {}
    for name, filename in MODEL_FILENAMES.items():
        file_path = os.path.join(MODEL_DIR, filename)
        # st.write(f"Loading {name} from {file_path}...") # Debugging line - uncomment for verbose loading
        loaded_assets[name] = load_model(file_path)
    st.success("All models and preprocessors loaded successfully!")
    return loaded_assets

# --- Prediction Function for Prospectivity Model ---
def predict_manganese_prospectivity_app(data_point: dict, preprocessor_obj, model_obj, numerical_features, categorical_features):
    if not isinstance(data_point, list):
        data_point = [data_point]
    input_df = pd.DataFrame(data_point)

    # Ensure all expected columns are present, fill with defaults if not provided
    expected_columns = numerical_features + categorical_features
    for col in categorical_features:
        if col not in input_df.columns:
            input_df[col] = 'Unknown'

    for col in numerical_features:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    # Reorder columns to match the training data
    input_df = input_df[expected_columns]

    processed_input = preprocessor_obj.transform(input_df)
    probabilities = model_obj.predict_proba(processed_input)[:, 1]
    return probabilities

# --- Hardcoded Unique Categories for Prospectivity Model (from original app.py) ---
unique_countries = ['Australia', 'Brazil', 'China', 'Gabon', 'India', 'Indonesia', 'Kazakhstan', 'Mexico', 'South Africa', 'Ukraine']
unique_states = ['Bihar', 'Goa', 'Karnataka', 'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Rajasthan', 'Uttar Pradesh', 'West Bengal']
unique_com_types = ['Carbonate', 'Oxide']
unique_oper_types = ['Open Pit', 'Surface', 'Underground']
unique_prod_sizes = ['L', 'M', 'S', 'XL']
unique_dev_stats = ['Advanced Exploration', 'Care & Maintenance', 'Deposit', 'Developer', 'Early Exploration', 'Feasibility', 'Mine', 'Past Producer', 'Producer', 'Reactivated', 'Reserve Base', 'Resource', 'Under Development']

# --- Global Variables/Thresholds for Reserve Estimation (assuming these were derived from original EDA) ---
ore_grade_col = 'Ore_Grade (%)'
tonnage_col = 'Tonnage'
ore_value_col = 'Ore_Value (¥/tonne)'
mining_cost_col = 'Mining_Cost (¥)'
processing_cost_col = 'Processing_Cost (¥)'
profit_col = 'Profit (¥)'
rock_type_col = 'Rock_Type'
waste_flag_col = 'Waste_Flag'
target_col = 'Target'

high_grade_threshold = 57.51  # Example value from notebook output
low_grade_threshold = 57.42   # Example value from notebook output

# Features for models (must match training features exactly)
features_task_a = ['X', 'Y', 'Z', tonnage_col, ore_value_col, mining_cost_col, processing_cost_col, rock_type_col, waste_flag_col, profit_col]
features_task_b = ['X', 'Y', 'Z', tonnage_col, ore_value_col, mining_cost_col, processing_cost_col, rock_type_col, waste_flag_col, profit_col, ore_grade_col]

# --- Prediction Function for Reserve Estimation Model ---
def estimate_reserve(input_df, pipeline_A, pipeline_B, high_grade_threshold, low_grade_threshold):
    if input_df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    # Add a dummy 'Block_ID' if not present for display purposes
    if 'Block_ID' not in input_df.columns:
        input_df['Block_ID'] = range(1, len(input_df) + 1)

    # Task A: Predict Ore Grade
    df_task_a = input_df[features_task_a].copy()
    
    input_df['Predicted_Ore_Grade'] = pipeline_A.predict(df_task_a)
    input_df[ore_grade_col] = input_df['Predicted_Ore_Grade']

    # Task B: Predict Economic Viability (probability)
    df_task_b = input_df[features_task_b].copy()
    input_df['Economic_Viability_Probability'] = pipeline_B.predict_proba(df_task_b)[:, 1]
    input_df['Predicted_Economic_Viability'] = (input_df['Economic_Viability_Probability'] > 0.5).astype(int) # Binary classification

    # Summarize results
    full_results_df = input_df.copy()
    economically_viable_blocks = full_results_df[full_results_df['Predicted_Economic_Viability'] == 1]

    estimated_tonnes = full_results_df[tonnage_col].sum()
    total_economically_viable_tonnes = economically_viable_blocks[tonnage_col].sum()
    average_predicted_grade_all_blocks = full_results_df['Predicted_Ore_Grade'].mean()
    average_predicted_grade_viable_blocks = economically_viable_blocks['Predicted_Ore_Grade'].mean() if not economically_viable_blocks.empty else 0
    economically_viable_blocks_count = len(economically_viable_blocks)
    high_grade_block_count = economically_viable_blocks[economically_viable_blocks['Predicted_Ore_Grade'] > high_grade_threshold].shape[0]
    low_grade_block_count = economically_viable_blocks[economically_viable_blocks['Predicted_Ore_Grade'] < low_grade_threshold].shape[0]

    return {
        'full_results_df': full_results_df,
        'estimated_tonnes': estimated_tonnes,
        'total_economically_viable_tonnes': total_economically_viable_tonnes,
        'average_predicted_grade_all_blocks': average_predicted_grade_all_blocks,
        'average_predicted_grade_viable_blocks': average_predicted_grade_viable_blocks,
        'economically_viable_blocks_count': economically_viable_blocks_count,
        'high_grade_block_count': high_grade_block_count,
        'low_grade_block_count': low_grade_block_count
    }

# --- Global Variables/Thresholds for Production Forecast ---
# From the original production forecast app.py
production_forecast_numerical_features = [
    'production_lag_1', 'production_lag_2', 'production_lag_3',
    'rolling_mean_3', 'rolling_std_3', 'growth_rate', 'volatility', 'year_int'
]

production_forecast_all_feature_columns = [
    'production_lag_1', 'production_lag_2', 'production_lag_3', 'rolling_mean_3', 'rolling_std_3', 'growth_rate', 'volatility', 'year_int',
    'country_Australia', 'country_Bolivia', 'country_Brazil', 'country_China', 'country_Colombia', 'country_Congo, D.R.', "country_Cote d'Ivoire",
    'country_Egypt', 'country_Gabon', 'country_Georgia', 'country_Ghana', 'country_Guyana', 'country_India', 'country_Indonesia', 'country_Iran',
    'country_Kazakhstan', 'country_Kenya', 'country_Malaysia', 'country_Mexico', 'country_Morocco', 'country_Myanmar', 'country_Namibia',
    'country_Nigeria', 'country_Oman', 'country_Pakistan', 'country_Peru', 'country_Romania', 'country_Russia', 'country_Russia, Europe',
    'country_Senegal', 'country_South Africa', 'country_Sudan', 'country_Thailand', 'country_Türkiye', 'country_Ukraine', 'country_Vietnam', 'country_Zambia'
]

production_forecast_country_options = [
    'Australia', 'Bolivia', 'Brazil', 'China', 'Colombia', 'Congo, D.R.', "Cote d'Ivoire",
    'Egypt', 'Gabon', 'Georgia', 'Ghana', 'Guyana', 'India', 'Indonesia', 'Iran',
    'Kazakhstan', 'Kenya', 'Malaysia', 'Mexico', 'Morocco', 'Myanmar', 'Namibia',
    'Nigeria', 'Oman', 'Pakistan', 'Peru', 'Romania', 'Russia', 'Russia, Europe',
    'Senegal', 'South Africa', 'Sudan', 'Thailand', 'Türkiye', 'Ukraine', 'Vietnam', 'Zambia'
]

# --- Global Variables/Thresholds for Shortfall Model ---
target_iron_concentrate = 65.0 # From notebook
risk_thresholds = {
    'LOW': (0, 0.30),
    'MEDIUM': (0.30, 0.60),
    'HIGH': (0.60, 0.80),
    'CRITICAL': (0.80, 1.01) # 1.01 to include 100%
}

# --- Helper Functions for Shortfall Model ---
@st.cache_data # Cache data loading for performance
def load_and_preprocess_historical_data(kaggle_dataset_path):
    """
    Loads and preprocesses the historical dataset to be used as context
    for feature engineering of new predictions.
    """
    try:
        df_hist = pd.read_csv(kaggle_dataset_path)
    except FileNotFoundError:
        st.error(f"Historical dataset not found at {kaggle_dataset_path}. Please ensure the file is present.")
        st.stop()

    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_hist = df_hist.set_index('date').sort_index()

    numeric_cols_to_convert = df_hist.select_dtypes(include='object').columns.tolist()
    for col in numeric_cols_to_convert:
        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')

    df_hist = df_hist.ffill().bfill() # Basic imputation
    return df_hist

def predict_shortfall_risk(
    model_obj, preprocessor_obj, x_train_cols,
    current_production, target_production, iron_feed, silica_feed,
    ore_pulp_ph, ore_pulp_density, f01_air, f02_air, f03_air, f04_air, f05_air, f06_air, f07_air,
    f01_level, f02_level, f03_level, f04_level, f05_level, f06_level, f07_level, risk_thresholds
):
    # Create a DataFrame for the new data point
    new_data = pd.DataFrame({
        'Flotation Column 01 Air Flow': [f01_air],
        'Flotation Column 02 Air Flow': [f02_air],
        'Flotation Column 03 Air Flow': [f03_air],
        'Flotation Column 04 Air Flow': [f04_air],
        'Flotation Column 05 Air Flow': [f05_air],
        'Flotation Column 06 Air Flow': [f06_air],
        'Flotation Column 07 Air Flow': [f07_air],
        'Flotation Column 01 Level': [f01_level],
        'Flotation Column 02 Level': [f02_level],
        'Flotation Column 03 Level': [f03_level],
        'Flotation Column 04 Level': [f04_level],
        'Flotation Column 05 Level': [f05_level],
        'Flotation Column 06 Level': [f06_level],
        'Flotation Column 07 Level': [f07_level],
        'Ore Pulp pH': [ore_pulp_ph],
        'Ore Pulp Density': [ore_pulp_density],
        'Iron Feed': [iron_feed],
        'Silica Feed': [silica_feed]
    })

    # Ensure all columns from training are present, fill missing with 0 or appropriate mean/median
    # This is critical if the model was trained on a specific set of features (e.g., from X_train_columns.pkl)
    for col in x_train_cols:
        if col not in new_data.columns:
            new_data[col] = 0.0 # Or some other sensible default
    new_data = new_data[x_train_cols] # Ensure column order

    processed_data = preprocessor_obj.transform(new_data)

    # Predict the probability of shortfall (class 1)
    probability = model_obj.predict_proba(processed_data)[0][1]

    risk_level = "Unknown"
    for level, (lower, upper) in risk_thresholds.items():
        if lower <= probability < upper:
            risk_level = level
            break

    # Calculate expected production and shortfall based on target iron concentrate
    # This is a placeholder, actual calculation might be more complex
    expected_production_ratio = target_production / target_iron_concentrate if target_iron_concentrate > 0 else 0
    expected_production = current_production * (1 - probability) # Simplified
    expected_shortfall = max(0, current_production - expected_production)

    return {
        'probability': probability,
        'risk_level': risk_level,
        'major_risk_factors': [], # SHAP not fully integrated here for brevity
        'target_production': target_production,
        'expected_production': expected_production,
        'expected_shortfall': expected_shortfall
    }

def plot_risk_gauge(probability, risk_level_str, target_production, expected_production, expected_shortfall):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100, # Convert to percentage
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Shortfall Risk Probability (%)", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen', 'name': 'Low Risk'},
                {'range': [30, 60], 'color': 'yellow', 'name': 'Medium Risk'},
                {'range': [60, 80], 'color': 'orange', 'name': 'High Risk'},
                {'range': [80, 100], 'color': 'red', 'name': 'Critical Risk'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=60, b=10),
        annotations=[
            dict(text=f"Risk Level: {risk_level_str}", x=0.5, y=0.1, font_size=14, showarrow=False, align='center'),
            dict(text=f"Expected Production: {expected_production:,.2f} tons", x=0.5, y=-0.05, font_size=12, showarrow=False, align='center'),
            dict(text=f"Expected Shortfall: {expected_shortfall:,.2f} tons", x=0.5, y=-0.15, font_size=12, showarrow=False, align='center')
        ]
    )
    return fig


# --- Streamlit App Structure ---
st.set_page_config(layout="wide", page_title="Unified Manganese AI Dashboard")

st.sidebar.title("Manganese AI Models")
selected_model = st.sidebar.radio(
    "Choose a Model",
    ["Prospectivity Prediction", "Reserve Estimation", "Production Forecast", "Production Shortfall Risk Predictor"]
)

st.title("Unified Manganese AI Dashboard")
st.write("Welcome to the Manganese AI System. Select a model from the sidebar to begin.")

# Load all models once at the start of the app
# This will execute only once due to @st.cache_resource
models = load_all_models()

# Placeholder sections for each model
if selected_model == "Prospectivity Prediction":
    st.header("Manganese Prospectivity Prediction Model")
    
    st.sidebar.header("Input Geological Parameters")

    # Input fields for Prospectivity Model
    latitude = st.sidebar.number_input('Latitude', min_value=-90.0, max_value=90.0, value=0.0, format="%.4f")
    longitude = st.sidebar.number_input('Longitude', min_value=-180.0, max_value=180.0, value=0.0, format="%.4f")
    
    country = st.sidebar.selectbox('Country', options=unique_countries, index=unique_countries.index('India') if 'India' in unique_countries else 0)
    state = st.sidebar.selectbox('State', options=unique_states, index=unique_states.index('Odisha') if 'Odisha' in unique_states else 0)
    com_type = st.sidebar.selectbox('Commodity Type', options=unique_com_types, index=unique_com_types.index('Oxide') if 'Oxide' in unique_com_types else 0)
    oper_type = st.sidebar.selectbox('Operation Type', options=unique_oper_types, index=unique_oper_types.index('Surface') if 'Surface' in unique_oper_types else 0)
    prod_size = st.sidebar.selectbox('Production Size', options=unique_prod_sizes, index=unique_prod_sizes.index('M') if 'M' in unique_prod_sizes else 0)
    dev_stat = st.sidebar.selectbox('Development Status', options=unique_dev_stats, index=unique_dev_stats.index('Producer') if 'Producer' in unique_dev_stats else 0)

    st.sidebar.markdown("### Ready to predict? 🚀")
    predict_button = st.sidebar.button('Predict Prospectivity!')

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

        numerical_features_for_pred = ['latitude', 'longitude']
        categorical_features_for_pred = ['country', 'state', 'com_type', 'oper_type', 'prod_size', 'dev_stat']

        prediction_probability = predict_manganese_prospectivity_app(
            user_input, 
            models["prospectivity_preprocessor"],
            models["prospectivity_model"],
            numerical_features_for_pred,
            categorical_features_for_pred
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

elif selected_model == "Reserve Estimation":
    st.header("Manganese Reserve Estimation Model ⛏️")
    st.markdown("### Estimate Ore Grade and Economic Viability for Manganese Blocks")

    input_df = pd.DataFrame()

    st.sidebar.header("Input Block Data")
    input_method = st.sidebar.radio("Choose input method:", ("Manual Input", "Upload CSV"))

    if input_method == "Manual Input":
        st.sidebar.subheader("Manual Block Data Entry")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x_coord = st.number_input('X Coordinate', value=0.0, format="%.2f")
            y_coord = st.number_input('Y Coordinate', value=0.0, format="%.2f")
            z_coord = st.number_input('Z Coordinate', value=0.0, format="%.2f")
            tonnage = st.number_input('Tonnage (tonnes)', min_value=0.0, value=1000.0, format="%.2f")
            ore_value = st.number_input('Ore Value (¥/tonne)', min_value=0.0, value=500.0, format="%.2f")
        with col2:
            mining_cost = st.number_input('Mining Cost (¥)', min_value=0.0, value=100.0, format="%.2f")
            processing_cost = st.number_input('Processing Cost (¥)', min_value=0.0, value=50.0, format="%.2f")
            profit = st.number_input('Profit (¥)', value=350.0, format="%.2f") # This would ideally be calculated, but can be input for model
            rock_type = st.selectbox('Rock Type', options=['Sedimentary', 'Metamorphic', 'Igneous', 'Other'], index=0)
            waste_flag = st.selectbox('Waste Flag', options=[0, 1], format_func=lambda x: 'Waste' if x==1 else 'Ore', index=0)

        manual_input_data = {
            'X': x_coord,
            'Y': y_coord,
            'Z': z_coord,
            tonnage_col: tonnage,
            ore_value_col: ore_value,
            mining_cost_col: mining_cost,
            processing_cost_col: processing_cost,
            profit_col: profit,
            rock_type_col: rock_type,
            waste_flag_col: waste_flag
        }
        input_df = pd.DataFrame([manual_input_data])
        predict_button_reserve = st.sidebar.button("Estimate Reserve (Manual) 🚀")

    else: # Upload CSV
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.sidebar.write("Uploaded data preview:")
            st.sidebar.write(input_df.head())
            predict_button_reserve = st.sidebar.button("Estimate Reserve from CSV 🚀")
        else:
            predict_button_reserve = False # No file, no button click

    # --- Prediction and Display Results ---
    if predict_button_reserve and not input_df.empty:
        st.subheader("Prediction Results ✨")
        
        try:
            reserve_summary = estimate_reserve(input_df, models["reserve_grade_pipeline"], models["reserve_viability_pipeline"], high_grade_threshold, low_grade_threshold)
            results_df = reserve_summary['full_results_df']

            st.markdown("#### Predicted Block Details:")
            st.dataframe(results_df[['Block_ID', 'X', 'Y', 'Z', tonnage_col, 
                                      'Predicted_Ore_Grade', 'Predicted_Economic_Viability', 
                                      'Economic_Viability_Probability']].round(2))
            
            st.markdown("#### Overall Reserve Summary 📊")
            st.write(f"**Total Input Tonnes:** {reserve_summary['estimated_tonnes']:,.2f} tonnes")
            st.write(f"**Total Economically Viable Tonnes:** {reserve_summary['total_economically_viable_tonnes']:,.2f} tonnes ✅")
            st.write(f"**Average Predicted Ore Grade (All Blocks):** {reserve_summary['average_predicted_grade_all_blocks']:.2f}% 🔬")
            st.write(f"**Average Predicted Ore Grade (Economically Viable Blocks):** {reserve_summary['average_predicted_grade_viable_blocks']:.2f}% 💎")
            st.write(f"**Number of Economically Viable Blocks:** {reserve_summary['economically_viable_blocks_count']} 💰")
            st.write(f"**Number of High-Grade Blocks (> {high_grade_threshold:.2f}%):** {reserve_summary['high_grade_block_count']} 🌟")
            st.write(f"**Number of Low-Grade Blocks (< {low_grade_threshold:.2f}%):** {reserve_summary['low_grade_block_count']} 📉")

        except ValueError as e:
            st.error(f"Input Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

    elif predict_button_reserve and input_df.empty:
        st.warning("Please provide input data, either manually or by uploading a CSV, to estimate reserves.")
    else:
        st.info("☝️ Use the sidebar to input block data or upload a CSV to get started!")

    st.markdown("--- ")
    st.markdown("**Note:** This model provides estimations based on historical data. Actual geological validation and drilling are always required for definitive reserve calculations. 🚧")

elif selected_model == "Production Forecast":
    st.header("Manganese Production Forecast Model")
    st.markdown("### Predict Future Manganese Production for Countries Worldwide! ✨")

    st.sidebar.header("Input Forecasting Parameters")

    selected_country = st.sidebar.selectbox(
        "Select Country",
        options=production_forecast_country_options,
        index=production_forecast_country_options.index('India') if 'India' in production_forecast_country_options else 0
    )

    # Year input
    current_year = pd.Timestamp.now().year
    year_int = st.sidebar.slider("Target Year", min_value=current_year, max_value=current_year + 10, value=current_year + 1)

    st.sidebar.markdown("--- ")
    st.sidebar.subheader("Historical Production Data (in tons)")
    st.sidebar.info("Provide recent production data to help forecast.")

    # Lagged production features
    production_lag_1 = st.sidebar.slider('Production Last Year', min_value=0.0, max_value=10000000.0, value=500000.0, step=1000.0)
    production_lag_2 = st.sidebar.slider('Production Two Years Ago', min_value=0.0, max_value=10000000.0, value=480000.0, step=1000.0)
    production_lag_3 = st.sidebar.slider('Production Three Years Ago', min_value=0.0, max_value=10000000.0, value=450000.0, step=1000.0)

    st.sidebar.markdown("--- ")
    st.sidebar.subheader("Derived Features (simulated for input)")
    st.sidebar.info("These are typically calculated from historical data. Adjust as needed for scenarios.")

    # Rolling window features and growth/volatility
    rolling_mean_3 = st.sidebar.slider('3-Year Rolling Mean Production', min_value=0.0, max_value=10000000.0, value=476666.0, step=1000.0)
    rolling_std_3 = st.sidebar.slider('3-Year Rolling Std Dev Production', min_value=0.0, max_value=1000000.0, value=25000.0, step=100.0)
    growth_rate = st.sidebar.slider('Annual Growth Rate (%)', min_value=-10.0, max_value=10.0, value=2.5, step=0.1, format="%.1f")
    volatility = st.sidebar.slider('Production Volatility Index', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    predict_button_forecast = st.sidebar.button("Forecast Production! 🔮")

    st.header('Forecast Results 📊')

    if predict_button_forecast:
        input_data = {
            'production_lag_1': production_lag_1,
            'production_lag_2': production_lag_2,
            'production_lag_3': production_lag_3,
            'rolling_mean_3': rolling_mean_3,
            'rolling_std_3': rolling_std_3,
            'growth_rate': growth_rate / 100.0, # Convert percentage to decimal
            'volatility': volatility,
            'year_int': year_int,
        }

        # Add one-hot encoded country features
        for country_col in production_forecast_country_options:
            feature_name = f'country_{country_col}'
            if feature_name in production_forecast_all_feature_columns: # Ensure the feature exists in the model's expected columns
                input_data[feature_name] = 1 if country_col == selected_country else 0
        
        # Ensure all features expected by the model are present, fill missing with 0
        for col in production_forecast_all_feature_columns:
            if col not in input_data:
                input_data[col] = 0 # Default to 0 for any missing feature

        input_df = pd.DataFrame([input_data])
        input_df = input_df[production_forecast_all_feature_columns] # Reorder columns to match model's training data

        try:
            prediction = models["production_forecast_model"].predict(input_df)[0]
            prediction_formatted = f"{prediction:,.2f}"
            st.success(f"The forecasted manganese production for **{selected_country}** in **{year_int}** is: ")
            st.balloons()
            st.markdown(f"## ➡️ {prediction_formatted} tons! 🌟")
            st.info("*(Note: Negative predictions are possible with Linear Regression if data is sparse, but XGBoost often handles this better. Consider post-processing for real-world application.)* 💡")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e} 💔")
            st.warning("Please ensure all input values are reasonable. 🙏")
    else:
        st.info("Adjust the parameters in the sidebar and click 'Forecast Production!' to see the results. 👆")

    st.markdown("--- ")
    st.markdown("Developed by ARAVIND INISH 🤖")

elif selected_model == "Production Shortfall Risk Predictor":
    st.header("Manganese Production Shortfall Risk Predictor")
    st.markdown("### Predict the Risk of Manganese Production Shortfall! ⚠️")

    st.sidebar.header("Shortfall Model Inputs")
    st.sidebar.info("Currently, these inputs are manual. We need to clarify how outputs from other models (Prospectivity, Reserve, Forecast) should influence these parameters.")

    # Define Kaggle Dataset Path for historical data loading within the app
    KAGGLE_DATASET_PATH = os.path.join(MODEL_DIR, 'MiningProcess_Flotation_Plant_Database.csv') # Assuming it's moved there
    # Temporarily create a dummy CSV if it doesn't exist for the Streamlit app to run without error
    if not os.path.exists(KAGGLE_DATASET_PATH):
        dummy_df = pd.DataFrame(np.random.rand(10, 20), columns=[f'col_{i}' for i in range(20)])
        dummy_df['date'] = pd.to_datetime(pd.date_range(start='2020-01-01', periods=10))
        dummy_df.to_csv(KAGGLE_DATASET_PATH, index=False)
        st.warning(f"Dummy file created at {KAGGLE_DATASET_PATH}. Please replace with actual data if needed.")

    historical_df = load_and_preprocess_historical_data(KAGGLE_DATASET_PATH)

    col1_sh, col2_sh = st.sidebar.columns(2)

    with col1_sh:
        current_production = st.number_input('Current Manganese Production (tons)', min_value=0.0, value=1000000.0, step=1000.0, format="%.2f")
        target_production = st.number_input('Target Iron Concentrate (%)', min_value=0.0, value=65.0, step=0.1, format="%.2f")
        iron_feed = st.number_input('Iron Feed', min_value=0.0, value=65.0, step=0.1, format="%.2f")
        silica_feed = st.number_input('Silica Feed', min_value=0.0, value=20.0, step=0.1, format="%.2f")
        ore_pulp_ph = st.number_input('Ore Pulp pH', min_value=0.0, max_value=14.0, value=9.0, step=0.1, format="%.2f")
        ore_pulp_density = st.number_input('Ore Pulp Density', min_value=0.0, value=1.7, step=0.01, format="%.2f")
    with col2_sh:
        st.markdown("**Flotation Column Air Flow**")
        f01_air = st.number_input('Column 01 Air Flow', min_value=0.0, value=170.0, step=1.0, format="%.1f")
        f02_air = st.number_input('Column 02 Air Flow', min_value=0.0, value=170.0, step=1.0, format="%.1f")
        f03_air = st.number_input('Column 03 Air Flow', min_value=0.0, value=170.0, step=1.0, format="%.1f")
        f04_air = st.number_input('Column 04 Air Flow', min_value=0.0, value=170.0, step=1.0, format="%.1f")
        f05_air = st.number_input('Column 05 Air Flow', min_value=0.0, value=170.0, step=1.0, format="%.1f")
        f06_air = st.number_input('Column 06 Air Flow', min_value=0.0, value=170.0, step=1.0, format="%.1f")
        f07_air = st.number_input('Column 07 Air Flow', min_value=0.0, value=170.0, step=1.0, format="%.1f")

        st.markdown("**Flotation Column Level**")
        f01_level = st.number_input('Column 01 Level', min_value=0.0, value=500.0, step=1.0, format="%.1f")
        f02_level = st.number_input('Column 02 Level', min_value=0.0, value=500.0, step=1.0, format="%.1f")
        f03_level = st.number_input('Column 03 Level', min_value=0.0, value=500.0, step=1.0, format="%.1f")
        f04_level = st.number_input('Column 04 Level', min_value=0.0, value=500.0, step=1.0, format="%.1f")
        f05_level = st.number_input('Column 05 Level', min_value=0.0, value=500.0, step=1.0, format="%.1f")
        f06_level = st.number_input('Column 06 Level', min_value=0.0, value=500.0, step=1.0, format="%.1f")
        f07_level = st.number_input('Column 07 Level', min_value=0.0, value=500.0, step=1.0, format="%.1f")

    predict_button_shortfall = st.sidebar.button("Predict Shortfall Risk! 🚨")

    st.header('Prediction Results 📊')

    if predict_button_shortfall:
        try:
            # Need to ensure 'x_train_cols' from models loaded by load_all_models
            # For now, it's loaded as models["shortfall_X_train_cols"]
            prediction_output = predict_shortfall_risk(
                model_obj=models["shortfall_model"],
                preprocessor_obj=models["shortfall_preprocessor"],
                x_train_cols=models["shortfall_X_train_cols"],
                current_production=current_production,
                target_production=target_production,
                iron_feed=iron_feed, silica_feed=silica_feed,
                ore_pulp_ph=ore_pulp_ph, ore_pulp_density=ore_pulp_density,
                f01_air=f01_air, f02_air=f02_air, f03_air=f03_air, f04_air=f04_air, f05_air=f05_air, f06_air=f06_air, f07_air=f07_air,
                f01_level=f01_level, f02_level=f02_level, f03_level=f03_level, f04_level=f04_level, f05_level=f05_level, f06_level=f06_level, f07_level=f07_level,
                risk_thresholds=risk_thresholds
            )

            if prediction_output:
                predicted_probability_float = float(prediction_output['probability'])

                st.subheader(f"Predicted Shortfall Probability: {predicted_probability_float:.2f} 📈")
                st.subheader(f"Risk Level: {prediction_output['risk_level']} {'🟢' if prediction_output['risk_level'] == 'LOW' else ('🟡' if prediction_output['risk_level'] == 'MEDIUM' else ('🟠' if prediction_output['risk_level'] == 'HIGH' else '🔴'))}")

                st.write(f"**Target Iron Concentrate:** {prediction_output['target_production']}% ")
                st.write(f"**Expected Production (based on current input & risk):** {prediction_output['expected_production']:,.2f} tons")
                st.write(f"**Potential Shortfall (based on current input & risk):** {prediction_output['expected_shortfall']:,.2f} tons")

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
                st.error("Prediction could not be generated. Please check input values.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e} 💔")
            st.warning("Please ensure all input values are reasonable. 🙏")
    else:
        st.info("Enter the parameters in the sidebar and click 'Predict Shortfall Risk' to see the results!")

    st.markdown("--- ")
    st.markdown("### ⚠️ **Important Clarification Needed** ⚠️")
    st.markdown("The Production Shortfall Risk Predictor model requires operational inputs like 'Iron Feed', 'Silica Feed', 'Ore Pulp pH', etc. Currently, these are manually entered.")
    st.markdown("**To achieve full integration as per the task requirements, we need clarification on how the outputs from the Prospectivity, Reserve Estimation, and Production Forecast models should be used to *derive* or *influence* these operational inputs for the Shortfall model.**")
    st.markdown("For example:")
    st.markdown("-   How does a 'High Prospectivity' or a 'High Reserve Grade' translate into expected 'Iron Feed' or 'Silica Feed' values for the Shortfall model?")
    st.markdown("-   How should the 'Forecasted Production' from the previous model be factored into the Shortfall model's operational parameters, beyond just the 'Current Manganese Production' input?")
    st.markdown("Please provide detailed guidance on these mappings for the next steps.")
    st.markdown("Developed for SIH26009 - Manganese Production Shortfall Risk Prediction Model (Proxy)")

