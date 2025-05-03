import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model("fogpredictionmodel.h5")

# Load dataset
df = pd.read_csv("testset.csv")

# Convert datetime column
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], format='%Y%m%d-%H:%M')
df['hour'] = df['datetime_utc'].dt.hour
df['day'] = df['datetime_utc'].dt.day
df['month'] = df['datetime_utc'].dt.month
df['year'] = df['datetime_utc'].dt.year

# Define features
features = [' _dewptm', ' _hum', ' _vism', ' _tempm', ' _pressurem', 'hour', 'day', 'month', 'year']
filtered_df = df[features + ['datetime_utc']].dropna()

# Normalize the features
scaler = MinMaxScaler()
filtered_df[features] = scaler.fit_transform(filtered_df[features])

# Prepare input data for prediction
X = filtered_df[features].values
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Predict fog density
filtered_df['fog_density'] = model.predict(X).flatten()

# Denormalize the fog density
filtered_df['fog_density_g_m3'] = filtered_df['fog_density'] * (0.5 - 0.05) + 0.05

# Denormalize the hour values (Fix)
filtered_df['hour'] = filtered_df['hour'] * (23 - 0) + 0
filtered_df['hour'] = filtered_df['hour'].round().astype(int)  # Ensure integer hours

# Group by denormalized hour
avg_fog_by_hour = filtered_df.groupby('hour')['fog_density_g_m3'].mean()

# Find the hour with the highest fog density
highest_fog_hour = avg_fog_by_hour.idxmax()
highest_fog_value = avg_fog_by_hour.max()

# Display results
st.title("Fog Density Analysis")
st.write(f"Based on historical data, fog is expected to be densest around **{highest_fog_hour}:00** "
         f"with an average fog density of **{highest_fog_value:.4f} g/m³**")

st.subheader("Fog Density by Hour")

# Create the figure for visualization
import streamlit as st
import plotly.graph_objects as go

# Create interactive figure
fig = go.Figure()

# Add scatter plot (interactive points)
fig.add_trace(go.Scatter(
    x=avg_fog_by_hour.index, 
    y=avg_fog_by_hour.values, 
    mode='lines+markers', 
    marker=dict(size=8, color='#A020F0', line=dict(width=2, color='white')), 
    line=dict(color='#A020F0', width=2.5),
    fill='tozeroy',  
    fillcolor='rgba(160, 32, 240, 0.2)',
    hoverinfo='x+y',  # Show both X and Y values on hover
    name="Fog Density"
))

# Set layout options
fig.update_layout(
    title="🌫 Fog Density by Hour",
    title_font=dict(size=16, color="white"),
    xaxis=dict(title="Hour of the Day", tickmode='linear', tick0=0, dtick=1),
    yaxis=dict(title="Average Fog Density (g/m³)"),
    template="plotly_dark",  # Dark background
    hovermode="x unified",  # Shows tooltip for all points on hover
)

# Display in Streamlit
st.plotly_chart(fig)

