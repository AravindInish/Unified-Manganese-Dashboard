
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration and Constants ---
st.set_page_config(page_title="Manganese Reserve Estimation ⛏️", layout="wide")

# --- Load Models and Preprocessors ---
@st.cache_resource
def load_models():
    try:
        pipeline_A = joblib.load('manganese_reserve_model_grade.pkl')
        pipeline_B = joblib.load('manganese_reserve_model_viability.pkl')
        # preprocessor_A = joblib.load('manganese_reserve_preprocessor_grade.pkl')
        # preprocessor_B = joblib.load('manganese_reserve_preprocessor_viability.pkl')
        return pipeline_A, pipeline_B
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'manganese_reserve_model_grade.pkl' and 'manganese_reserve_model_viability.pkl' are in the same directory.")
        st.stop()

pipeline_A, pipeline_B = load_models()

# --- Global Variables/Thresholds (assuming these were derived from original EDA) ---
# These would ideally be saved with the model or derived dynamically if data allows.
# For this example, using representative values based on the notebook's output.
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
features_task_b = ['X', 'Y', 'Z', tonnage_col, ore_grade_col, ore_value_col, mining_cost_col, processing_cost_col, rock_type_col, profit_col]

# --- Reserve Estimation Function (re-used from notebook) ---
def estimate_reserve(block_data_df: pd.DataFrame):
    if block_data_df.empty:
        return {
            'estimated_tonnes': 0,
            'total_predicted_ore_tonnes': 0,
            'total_economically_viable_tonnes': 0,
            'average_predicted_grade_all_blocks': 0,
            'average_predicted_grade_viable_blocks': 0,
            'economically_viable_blocks_count': 0,
            'high_grade_block_count': 0,
            'low_grade_block_count': 0,
            'predicted_grades': pd.Series(),
            'economic_viabilities': pd.Series(),
            'confidence_indicator_probabilities': pd.Series()
        }

    # Make predictions using the trained pipelines
    predicted_grades = pipeline_A.predict(block_data_df[features_task_a])

    block_data_df_with_pred_grade = block_data_df.copy()
    block_data_df_with_pred_grade[ore_grade_col] = predicted_grades

    economic_viabilities = pipeline_B.predict(block_data_df_with_pred_grade[features_task_b])
    confidence_indicator_probabilities = pipeline_B.predict_proba(block_data_df_with_pred_grade[features_task_b])[:, 1]

    block_data_df_results = block_data_df.copy()
    block_data_df_results['Predicted_Ore_Grade'] = predicted_grades
    block_data_df_results['Predicted_Economic_Viability'] = economic_viabilities
    block_data_df_results['Economic_Viability_Probability'] = confidence_indicator_probabilities

    total_estimated_tonnes = block_data_df_results[tonnage_col].sum()
    economically_viable_blocks_results = block_data_df_results[block_data_df_results['Predicted_Economic_Viability'] == 1]
    total_economically_viable_tonnes = economically_viable_blocks_results[tonnage_col].sum()

    average_predicted_grade_all_blocks = block_data_df_results['Predicted_Ore_Grade'].mean()
    average_predicted_grade_viable_blocks = economically_viable_blocks_results['Predicted_Ore_Grade'].mean() if not economically_viable_blocks_results.empty else 0

    economically_viable_blocks_count = economically_viable_blocks_results.shape[0]

    high_grade_block_count_func = block_data_df_results[block_data_df_results['Predicted_Ore_Grade'] > high_grade_threshold].shape[0]
    low_grade_block_count_func = block_data_df_results[block_data_df_results['Predicted_Ore_Grade'] < low_grade_threshold].shape[0]

    return {
        'estimated_tonnes': total_estimated_tonnes,
        'total_predicted_ore_tonnes': total_estimated_tonnes, # This is the sum of tonnage in the input blocks
        'total_economically_viable_tonnes': total_economically_viable_tonnes,
        'average_predicted_grade_all_blocks': average_predicted_grade_all_blocks,
        'average_predicted_grade_viable_blocks': average_predicted_grade_viable_blocks,
        'economically_viable_blocks_count': economically_viable_blocks_count,
        'high_grade_block_count': high_grade_block_count_func,
        'low_grade_block_count': low_grade_block_count_func,
        'predicted_grades': predicted_grades,
        'economic_viabilities': economic_viabilities,
        'confidence_indicator_probabilities': confidence_indicator_probabilities,
        'full_results_df': block_data_df_results # Added for detailed output
    }


# --- Streamlit UI ---
st.title("Manganese Reserve Estimation Model 📈")
st.markdown("Estimate the quantity/value of potentially recoverable ore from mining block data.")

st.sidebar.header("Input Block Data 📝")
st.sidebar.markdown("Enter parameters for a single mining block or upload a CSV for multiple blocks.")

# Input method selection
input_method = st.sidebar.radio("Choose Input Method:", ("Single Block Input", "Upload CSV"))

input_df = pd.DataFrame()

if input_method == "Single Block Input":
    with st.sidebar.form("single_block_form"):
        st.subheader("Single Block Parameters")
        col1, col2, col3 = st.columns(3)
        block_id = col1.text_input("Block ID", "B_New_001")
        x = col1.number_input("X Coordinate", value=250, step=10)
        y = col2.number_input("Y Coordinate", value=250, step=10)
        z = col3.number_input("Z Coordinate", value=50, step=5)
        
        tonnage = st.number_input("Tonnage (tonnes)", value=2000, step=100, min_value=1)
        rock_type = st.selectbox("Rock Type", ['Magnetite', 'Hematite', 'Waste'], index=0)
        
        # These are used as features, not targets to be predicted by user
        ore_value = st.number_input("Ore Value (¥/tonne)", value=300.0, step=10.0)
        mining_cost = st.number_input("Mining Cost (¥)", value=45, step=1)
        processing_cost = st.number_input("Processing Cost (¥)", value=25.0, step=1.0)
        waste_flag = st.selectbox("Waste Flag (1 if waste, 0 if not)", [0, 1], index=0)
        profit = st.number_input("Profit (¥)", value=400000.0, step=10000.0)
        
        submitted = st.form_submit_button("Estimate Reserve for Single Block 🚀")
        
        if submitted:
            input_data = {
                'Block_ID': block_id,
                'X': x, 'Y': y, 'Z': z,
                'Rock_Type': rock_type,
                'Tonnage': tonnage,
                'Ore_Value (¥/tonne)': ore_value,
                'Mining_Cost (¥)': mining_cost,
                'Processing_Cost (¥)': processing_cost,
                'Waste_Flag': waste_flag,
                'Profit (¥)': profit,
            }
            input_df = pd.DataFrame([input_data])
            
elif input_method == "Upload CSV":
    st.sidebar.subheader("Upload Block Data CSV")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.sidebar.write("Uploaded data preview:")
        st.sidebar.write(input_df.head())
        if st.sidebar.button("Estimate Reserve from CSV 🚀"):
            pass # This pass will allow the code below to run with the loaded df

# --- Prediction and Display Results ---
if not input_df.empty:
    st.subheader("Prediction Results ✨")
    
    try:
        reserve_summary = estimate_reserve(input_df)
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

else:
    st.info("☝️ Use the sidebar to input block data or upload a CSV to get started!")

st.markdown("--- ")
st.markdown("**Note:** This model provides estimations based on historical data. Actual geological validation and drilling are always required for definitive reserve calculations. 🚧")
