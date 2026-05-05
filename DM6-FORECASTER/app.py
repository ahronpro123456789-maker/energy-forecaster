import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Energy Demand Forecaster", page_icon="⚡", layout="wide")

st.title("Electricity Consumption Forecasting and Anomaly Detection")
st.markdown("""
Predicting future energy demand and identifying abnormal usage patterns to support efficient energy management and optimization.
""")
st.info("**Developed by:** John Franklin Bugauisan, William Ray Respicio, Ahron John Barlis, Carlo Rossi Gallardo")
st.markdown("---")


# ── Helper Functions ───────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


@st.cache_data
def resample_data(values, index_strs, freq):
    """Resample a series to the chosen frequency. Handles duplicate timestamps."""
    s = pd.Series(values, index=pd.to_datetime(index_strs))
    s = s.sort_index()
    # Deduplicate timestamps by averaging duplicates
    s = s.groupby(s.index).mean()
    if freq == "Hourly (original)":
        return s
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    return s.resample(freq_map[freq]).mean()


def adf_test(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return result[1]          # p-value


def auto_diff(series, max_d=2):
    """Return the minimum d that makes the series stationary (ADF p<0.05)."""
    for d in range(0, max_d + 1):
        s = series.diff(d).dropna() if d > 0 else series
        if adf_test(s) < 0.05:
            return d
    return max_d


def suggest_arima_orders(series, max_p=5, max_q=5):
    """
    Suggest p, d, q from ADF + ACF/PACF analysis.
    Uses a simple BIC grid search over a small candidate set.
    """
    d = auto_diff(series)
    diff_series = series.diff(d).dropna() if d > 0 else series

    # Candidate p from PACF, q from ACF (look at first significant lag)
    try:
        pacf_vals = pacf(diff_series, nlags=max_p, method='ywm')
        acf_vals  = acf(diff_series,  nlags=max_q)
        sig = 1.96 / np.sqrt(len(diff_series))
        p_cand = [i for i in range(1, max_p+1) if abs(pacf_vals[i]) > sig]
        q_cand = [i for i in range(1, max_q+1) if abs(acf_vals[i])  > sig]
        p_cand = p_cand[:3] if p_cand else [1]
        q_cand = q_cand[:3] if q_cand else [1]
    except Exception:
        p_cand, q_cand = [1], [1]

    best_bic, best_order = np.inf, (p_cand[0], d, q_cand[0])
    for p in p_cand:
        for q in q_cand:
            try:
                bic = ARIMA(series, order=(p, d, q)).fit().bic
                if bic < best_bic:
                    best_bic, best_order = bic, (p, d, q)
            except Exception:
                continue
    return best_order


def calculate_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) /
                           np.clip(np.abs(y_true), 1e-8, None))) * 100
    return mae, rmse, mape


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Data Setup")
uploaded_file = st.sidebar.file_uploader("Upload Time Series CSV", type=['csv'])

if uploaded_file is None:
    st.warning("Please upload your dataset in the sidebar to begin.")
    st.stop()

df_raw = load_data(uploaded_file)

# Row limit option
st.sidebar.markdown("---")
st.sidebar.subheader("Data Sampling")
use_sample = st.sidebar.checkbox("Limit rows (for faster processing)", value=False)
if use_sample:
    max_rows = st.sidebar.number_input(
        "Number of rows to use:", 
        min_value=1000, 
        max_value=len(df_raw), 
        value=min(10000, len(df_raw)),
        step=1000,
        help="Using fewer rows will speed up processing"
    )
    df_raw = df_raw.head(max_rows)
    st.sidebar.info(f"Using first {len(df_raw):,} rows")

st.sidebar.markdown("---")
datetime_col = st.sidebar.selectbox("Datetime Column:", df_raw.columns, index=0)
numeric_cols  = df_raw.select_dtypes(include=np.number).columns.tolist()
target_col    = st.sidebar.selectbox("Target Variable:", numeric_cols)

try:
    df_raw[datetime_col] = pd.to_datetime(df_raw[datetime_col])
    df_raw.set_index(datetime_col, inplace=True)
    df_raw.sort_index(inplace=True)
    st.sidebar.success(f"{len(df_raw):,} rows loaded.")
except Exception as e:
    st.sidebar.error(f"Datetime error: {e}")
    st.stop()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Data Overview",
    "Forecasting",
    "Anomaly Detection"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW (SIMPLIFIED)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data Overview")
    
    # Simple date filter only
    col1, col2 = st.columns([2, 1])
    with col1:
        min_date = df_raw.index.min().date()
        max_date = df_raw.index.max().date()
        start_date, end_date = st.date_input(
            "Date Range:",
            [min_date, max_date],
            min_value=min_date, max_value=max_date
        )
    with col2:
        st.metric("Total Rows", f"{len(df_raw):,}")

    # Filter data
    mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
    filtered_df = df_raw.loc[mask]

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    # Historical trend graph
    st.subheader("Historical Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(filtered_df.index, filtered_df[target_col], color='steelblue', linewidth=0.8)
    ax.set_ylabel("Consumption (MW)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


    st.subheader("Historical Trend")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(filtered_df.index, filtered_df[target_col], color='steelblue', linewidth=0.4)
    ax.set_title(f"{target_col} — Full Historical Trend")
    ax.set_ylabel("Consumption (MW)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Basic stats
    st.subheader("Descriptive Statistics")
    st.dataframe(filtered_df[target_col].describe().to_frame().T, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECASTING (SIMPLIFIED)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("ARIMA Forecasting")
    
    with st.expander("What is ARIMA?", expanded=False):
        st.markdown("""
        **ARIMA** predicts future values based on past patterns in the data.
        - Uses historical consumption to forecast future demand
        - Provides confidence intervals showing prediction uncertainty
        - Ideal for short to medium-term energy forecasting
        """)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Configuration")

        freq_choice = st.selectbox(
            "Data Frequency:",
            ["Daily", "Weekly", "Monthly"],
            index=0
        )

        train_pct = st.slider("Training Data (%)", 50, 95, 80)
        
        st.markdown("")
        run_btn = st.button("Run Forecast", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            # Prepare data
            full_series = df_raw[target_col].dropna()
            resampled = resample_data(
                full_series.values,
                full_series.index.astype(str).tolist(),
                freq_choice
            )
            resampled = resampled.dropna()

            split_idx = int(len(resampled) * (train_pct / 100))
            train_data = resampled.iloc[:split_idx]
            test_data = resampled.iloc[split_idx:]

            # Auto-select ARIMA orders (hidden from user)
            with st.spinner("Training ARIMA model..."):
                best_p, best_d, best_q = suggest_arima_orders(train_data)
                p, d, q = best_p, best_d, best_q

                # Fit model
                try:
                    fitted = ARIMA(train_data, order=(p, d, q)).fit()

                    # Forecast
                    fc_obj = fitted.get_forecast(steps=len(test_data))
                    predictions = fc_obj.predicted_mean
                    conf_int = fc_obj.conf_int(alpha=0.05)
                    predictions.index = test_data.index
                    conf_int.index = test_data.index

                    mae, rmse, mape = calculate_metrics(test_data.values, predictions.values)

                    # Metrics
                    st.subheader("Performance Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"{mae:.2f} MW")
                    m2.metric("RMSE", f"{rmse:.2f} MW")
                    m3.metric("MAPE", f"{mape:.2f}%")

                    # Forecast plot
                    st.subheader("Forecast Results")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    
                    # Plot last 200 training points for context
                    plot_train = train_data.tail(200)
                    ax.plot(plot_train.index, plot_train,
                            label='Historical', color='gray', linewidth=1, alpha=0.6)
                    
                    # Plot actual test data
                    ax.plot(test_data.index, test_data,
                            label='Actual', color='green', linewidth=1.5)
                    
                    # Plot forecast
                    ax.plot(test_data.index, predictions,
                            label='Forecast', color='red', linestyle='--', linewidth=1.5)
                    
                    # Plot confidence interval
                    ax.fill_between(
                        test_data.index,
                        conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        alpha=0.2, color='red', label='95% Confidence'
                    )
                    
                    ax.set_title(f"ARIMA Forecast ({freq_choice})", fontsize=13, pad=15)
                    ax.set_ylabel("Consumption (MW)")
                    ax.set_xlabel("Date")
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Download button
                    result_df = pd.DataFrame({
                        "Actual": test_data.values,
                        "Forecast": predictions.values.round(2),
                        "Error": (test_data.values - predictions.values).round(2)
                    }, index=test_data.index)
                    
                    csv_forecast = result_df.reset_index().to_csv(index=False)
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv_forecast,
                        file_name=f"forecast_{freq_choice.lower()}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Model fitting failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION (SIMPLIFIED)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Anomaly Detection")
    st.header("Anomaly Detection")
    
    with st.expander("What is Anomaly Detection?", expanded=False):
        st.markdown("""
        **Anomaly Detection** identifies unusual consumption patterns that deviate from normal behavior.
        - Uses Z-score method: measures how far each point is from the average
        - Helps identify equipment issues, demand spikes, or unusual events
        - Threshold of 3σ means "3 standard deviations from normal"
        """)

    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuration")
        
        z_threshold = st.slider(
            "Z-score Threshold:", 
            2.0, 4.0, 3.0, step=0.1,
            help="Higher = stricter (fewer anomalies)"
        )
        
        st.markdown("")
        run_anomaly = st.button("Run Detection", type="primary", use_container_width=True)

    with col2:
        if run_anomaly:
            with st.spinner("Detecting anomalies..."):
                # Calculate Z-scores
                mean_val = df_raw[target_col].mean()
                std_val = df_raw[target_col].std()
                z_scores = np.abs((df_raw[target_col] - mean_val) / std_val)
                
                # Find anomalies
                anomaly_mask = z_scores > z_threshold
                anomalies = df_raw[anomaly_mask]

                # Results
                st.subheader("Detection Results")
                st.metric("Total Anomalies Detected", f"{len(anomalies):,}")

                # Anomaly plot
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_raw.index, df_raw[target_col],
                        label='Normal', color='steelblue', alpha=0.5, linewidth=0.5)
                ax.scatter(anomalies.index, anomalies[target_col],
                           color='red', label=f'Anomalies ({len(anomalies):,})',
                           zorder=5, s=15, alpha=0.8)
                ax.set_title(f"Anomaly Detection (Threshold: {z_threshold}σ)", fontsize=13, pad=15)
                ax.set_ylabel("Consumption (MW)")
                ax.set_xlabel("Date")
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                # Download button
                anomaly_df = pd.DataFrame({
                    "Consumption": anomalies[target_col].values,
                    "Z_Score": z_scores[anomaly_mask].values.round(2)
                }, index=anomalies.index)
                
                csv_anomalies = anomaly_df.reset_index().to_csv(index=False)
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv_anomalies,
                    file_name=f"anomalies_threshold_{z_threshold}.csv",
                    mime="text/csv"
                )
