import streamlit as st
import joblib
import numpy as np
import sys

# Configure Streamlit to run on port 8502
if __name__ == '__main__':
    sys.argv = ["streamlit", "run", __file__, "--server.port", "8502"]

# Load the trained XGBoost model
model = joblib.load('xgboost_model.joblib')

st.title("💧 Water Quality Predictor")

st.markdown("""
This app predicts whether the water is **safe to drink** based on the following parameters:
- pH level
- Hardness
- Solids
- Turbidity
""")

# Input sliders
ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.slider("Hardness", min_value=0.0, max_value=500.0, value=234.27, step=0.1)
solids = st.slider("Solids", min_value=0.0, max_value=35000.0, value=26811.20, step=0.1)
turbidity = st.slider("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=3.52, step=0.1)

# Calculate derived features
ph_deviation = abs(ph - 7.0)  # Deviation from ideal pH (7.0)
log_turbidity = np.log1p(turbidity)  # Log transform of turbidity
tds_hardness_ratio = solids / hardness if hardness != 0 else 0
estimated_hardness = solids / 10  # Simple estimation
ph_turbidity_ratio = ph / turbidity if turbidity != 0 else 0
hardness_squared = hardness ** 2
solids_turbidity_product = solids * turbidity
ph_hardness_interaction = ph * hardness
ph_squared = ph ** 2
solids_squared = solids ** 2
turbidity_squared = turbidity ** 2
ph_ideal_distance = abs(ph - 7.0)  # Distance from ideal pH
turbidity_ideal_distance = abs(turbidity - 1.0)  # Distance from ideal turbidity
ph_solids_hardness = ph * solids * hardness

# Prediction
if st.button("Predict Water Potability"):
    features = np.array([[
        ph, hardness, solids, turbidity,
        ph_deviation, log_turbidity, tds_hardness_ratio,
        estimated_hardness, ph_turbidity_ratio, hardness_squared,
        solids_turbidity_product, ph_hardness_interaction,
        ph_squared, solids_squared, turbidity_squared,
        ph_ideal_distance, turbidity_ideal_distance,
        ph_solids_hardness
    ]])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    if prediction == 0:
        st.error("❌ The water is **Not Drinkable**")
    elif prediction == 1:
        st.warning("⚠️ The water is **Usable**")
    else:
        st.success("✅ The water is **Drinkable**")
    
    # Display probabilities
    st.write("Prediction Probabilities:")
    st.write(f"- Not Drinkable: {probabilities[0]:.2%}")
    st.write(f"- Usable: {probabilities[1]:.2%}")
    st.write(f"- Drinkable: {probabilities[2]:.2%}")
