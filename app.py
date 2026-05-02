import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Energy Demand Forecaster", page_icon="⚡", layout="wide")

# --- Header & Group Info ---
st.title("⚡ Electricity Consumption Forecasting and Anomaly Detection")
st.markdown("""
**CS DM 6-DATA MINING METHODOLOGIES: Final Requirement Case Study**  
Predicting future energy demand and identifying abnormal usage patterns to support efficient energy management and optimization.
""")
st.info("**Developed by:** John Franklin Bugauisan, William Ray Respicio, Ahron John Barlis, Carlo Rossi Gallardo")
st.markdown("---")

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    # Loads the uploaded file
    df = pd.read_csv(file)
    return df

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# --- Sidebar: File Upload & Config ---
st.sidebar.header("📁 Data Setup")
uploaded_file = st.sidebar.file_uploader("Upload Time Series Dataset (CSV)", type=['csv'])

if uploaded_file is None:
    st.warning("👈 Please upload your dataset (e.g., 'PJME_engineered.csv') in the sidebar to begin.")
    st.stop()

# Load Data
df = load_data(uploaded_file)

# Select Columns
datetime_col = st.sidebar.selectbox("Select Datetime Column:", df.columns, index=0)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
target_col = st.sidebar.selectbox("Select Target Variable (Consumption):", numeric_cols)

# Process Datetime
try:
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    df.sort_index(inplace=True)
    st.sidebar.success("Dataset loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error parsing datetime: {e}")
    st.stop()


# --- Main App: Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Data Overview & Filters", "📈 Time Series Forecasting", "🔍 Anomaly Detection"])

# ==========================================
# TAB 1: DATA OVERVIEW & FILTERS
# ==========================================
with tab1:
    st.header("Data Preview and Advanced Filtering")
    st.write("Explore your dataset before applying machine learning models.")
    
    # Advanced Filters
    col1, col2 = st.columns(2)
    with col1:
        # Date Range Filter
        min_date = df.index.min().date()
        max_date = df.index.max().date() if pd.notnull(df.index.max()) else min_date
            
        start_date, end_date = st.date_input(
            "Filter by Date Range:",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        # Value Range Filter
        min_val = float(df[target_col].min())
        max_val = float(df[target_col].max())
        val_range = st.slider(
            f"Filter by {target_col} Range:",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )

    # Apply Filters
    mask = (df.index.date >= start_date) & (df.index.date <= end_date) & \
           (df[target_col] >= val_range[0]) & (df[target_col] <= val_range[1])
    filtered_df = df.loc[mask]
    
    st.success(f"Showing {len(filtered_df)} rows based on current filters.")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Plot Raw Data
    st.subheader("Historical Trend (Filtered Data)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(filtered_df.index, filtered_df[target_col], color='blue', linewidth=0.5)
    ax.set_title(f"{target_col} over time")
    ax.set_ylabel("Consumption")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ==========================================
# TAB 2: TIME SERIES FORECASTING
# ==========================================
with tab2:
    st.header("📈 Time Series Forecasting (ARIMA)")
    st.write("Using Historical Data to predict future electricity demand.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Forecasting Parameters")
        forecast_steps = st.number_input("Forecast Horizon (Steps):", min_value=1, max_value=365, value=30)
        train_size = st.slider("Training Data Size (%)", 50, 95, 80)
        run_forecast = st.button("Run Forecasting Model", type="primary")
        
    with col2:
        if run_forecast:
            with st.spinner("Training ARIMA model on the ENTIRE dataset..."):
                # Train/Test Split
                data = df[target_col].dropna()
                split_idx = int(len(data) * (train_size / 100))
                train, test = data.iloc[:split_idx], data.iloc[split_idx:]
                
                # Fit Model
                model = ARIMA(train, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Predict and get Confidence Intervals
                forecast = fitted_model.get_forecast(steps=len(test))
                predictions = forecast.predicted_mean
                conf_int = forecast.conf_int()
                
                # Metrics
                mae, rmse = calculate_metrics(test, predictions)
                
                # Display Metrics
                m1, m2 = st.columns(2)
                m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Note: We only plot the last 500 points of the training set so the graph 
                # doesn't look like a squished block of ink, but the model trained on all of it!
                plot_train = train.tail(500) 
                ax.plot(plot_train.index, plot_train, label='Training Data (Last 500 displayed)')
                ax.plot(test.index, test, label='Actual Test Data', color='green')
                ax.plot(test.index, predictions, label='Forecast', color='red', linestyle='--')
                
                # Confidence Interval
                ax.fill_between(test.index, 
                                conf_int.iloc[:, 0], 
                                conf_int.iloc[:, 1], 
                                color='red', alpha=0.15, label='Confidence Interval')
                
                ax.set_title("Electricity Consumption Forecast")
                ax.set_ylabel("Consumption")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

# ==========================================
# TAB 3: ANOMALY DETECTION
# ==========================================
with tab3:
    st.header("🔍 Anomaly Detection")
    st.write("Identifying abnormal usage patterns indicating potential equipment failure or reporting errors.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Anomaly Detection Parameters")
        contamination = st.slider("Contamination (Expected Anomaly %)", 0.01, 0.10, 0.05, step=0.01)
        run_anomaly = st.button("Run Anomaly Detection", type="primary")
        
    with col2:
        if run_anomaly:
            with st.spinner("Detecting Outliers using Isolation Forest on the ENTIRE dataset..."):
                data = df[[target_col]].dropna().copy()
                
                # Fit Isolation Forest
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                data['Anomaly'] = iso_forest.fit_predict(data[[target_col]])
                
                # -1 indicates anomaly, 1 indicates normal
                anomalies = data[data['Anomaly'] == -1]
                
                st.metric("Total Anomalies Detected", len(anomalies))
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data[target_col], label='Normal Consumption', color='blue', alpha=0.6, linewidth=0.5)
                ax.scatter(anomalies.index, anomalies[target_col], color='red', label='Anomaly', zorder=5, s=20)
                ax.set_title("Outlier Detection in Electricity Consumption")
                ax.set_ylabel("Consumption")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
                
                # Show anomaly timestamps
                st.subheader("View Anomaly Timestamps")
                st.dataframe(anomalies[[target_col]].sort_index(ascending=False), use_container_width=True)