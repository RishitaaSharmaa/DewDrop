# Fog Prediction Model

## Overview
The **Fog Prediction Model** is a deep learning-based system that utilizes **Long Short-Term Memory (LSTM) networks** to predict fog conditions based on historical weather data. The model helps in forecasting fog occurrences, aiding transportation safety and logistics planning.

## Features
- **Deep Learning-based Forecasting**: Uses LSTM for time-series prediction.
- **Historical Data Analysis**: Predicts fog conditions based on past weather data.
- **Web-Based Interface**: Built using Flask for easy interaction.
- **Scalable**: Can be trained with different datasets for enhanced accuracy.

## Dataset
The model is trained using weather datasets containing features such as:
- **Temperature**
- **Humidity**
- **Dew Point**
- **Visibility**
- **Wind Speed**

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **Flask** (for web-based UI)
- **Pandas** (for data manipulation)
- **NumPy**
- **Matplotlib** (for visualization)
- **Scikit-learn** (for data preprocessing)

## Model Architecture
The LSTM model consists of:
1. **LSTM Layers**: Captures sequential dependencies in weather data.
2. **Dense Layers**: Processes extracted features for final predictions.
3. **Activation Functions**: ReLU for hidden layers, Sigmoid/Softmax for output.
4. **Optimization Algorithm**: Adam optimizer for efficient training.

## Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow/Keras
- Flask
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Future Enhancements
- Improve dataset variety for better generalization.
- Deploy model using cloud-based services for real-time forecasting.




