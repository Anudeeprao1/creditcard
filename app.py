import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Function to process data and detect anomalies
def process_data(file):
    # Load data from the uploaded file
    data = pd.read_excel(file, sheet_name='creditcard_test')

    # Identify object columns
    object_columns = data.select_dtypes(include=['object']).columns
    st.write(f"Object columns: {object_columns}")

    # Convert object columns to float, and drop rows with conversion errors
    for column in object_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Drop rows with any NaN values
    data.dropna(inplace=True)

    # Load scaler and model
    scaler = StandardScaler()
    scaler.fit(data)  # Fit scaler on data (this is typically done during training, here it's for demonstration)
    model = tf.keras.models.load_model('autoencoder_model.keras', compile=False)

    # Prepare features
    X = data.values

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    reconstructed = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

    # Calculate threshold as the 95th percentile of the reconstruction errors
    threshold = np.percentile(mse, 95)
    st.write(f"Calculated threshold: {threshold}")

    # Determine anomalies based on calculated threshold
    anomalies = mse > threshold

    # Create a DataFrame to hold results
    results = data.copy()
    results['MSE'] = mse
    results['Anomaly'] = anomalies.astype(int)  # 1 for anomalies, 0 for normal

    return results, threshold

# Streamlit UI
st.title('Anomaly Detection with Autoencoder')

# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Process the file
    results, threshold = process_data(uploaded_file)

    # Display the data
    st.write("Data preview:")
    st.write(results.head())

    # Visualization
    st.write("Distribution of MSE with Anomaly Threshold")
    fig, ax = plt.subplots()
    sns.histplot(results['MSE'], bins=50, kde=True, ax=ax)
    ax.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    ax.set_title('Distribution of MSE with Anomaly Threshold')
    ax.set_xlabel('Mean Squared Error (MSE)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    st.write("Anomalies vs. Normal Points")
    fig, ax = plt.subplots()
    ax.scatter(range(len(results)), results['MSE'], c=results['Anomaly'], cmap='coolwarm', s=10)
    ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    ax.set_title('Anomalies vs. Normal Points')
    ax.set_xlabel('Index')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.legend()
    st.pyplot(fig)
    
# Remove MSE column
    results = results.drop(columns=['MSE'], errors='ignore')

    # Provide download option
    @st.cache
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(results)
    st.download_button(
        label="Download Processed Data",
        data=csv_data,
        file_name='processed_data.csv',
        mime='text/csv'
    )
