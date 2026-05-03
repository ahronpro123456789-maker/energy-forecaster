import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Energy Demand Forecaster", page_icon="⚡", layout="wide")

st.title("⚡ Electricity Consumption Forecasting and Anomaly Detection")
st.markdown("""
**CS DM 6-DATA MINING METHODOLOGIES: Final Requirement Case Study**  
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
st.sidebar.header("📁 Data Setup")
uploaded_file = st.sidebar.file_uploader("Upload Time Series CSV", type=['csv'])

if uploaded_file is None:
    st.warning("👈 Please upload your dataset (e.g., 'PJME_engineered.csv') in the sidebar to begin.")
    st.stop()

df_raw = load_data(uploaded_file)

datetime_col = st.sidebar.selectbox("Datetime Column:", df_raw.columns, index=0)
numeric_cols  = df_raw.select_dtypes(include=np.number).columns.tolist()
target_col    = st.sidebar.selectbox("Target Variable:", numeric_cols)

try:
    df_raw[datetime_col] = pd.to_datetime(df_raw[datetime_col])
    df_raw.set_index(datetime_col, inplace=True)
    df_raw.sort_index(inplace=True)
    st.sidebar.success(f"✅ {len(df_raw):,} rows loaded.")
except Exception as e:
    st.sidebar.error(f"Datetime error: {e}")
    st.stop()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Data Overview & Filters",
    "📈 Time Series Forecasting",
    "🔍 Anomaly Detection"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data Preview and Advanced Filtering")

    col1, col2 = st.columns(2)
    with col1:
        min_date = df_raw.index.min().date()
        max_date = df_raw.index.max().date()
        start_date, end_date = st.date_input(
            "Filter by Date Range:",
            [min_date, max_date],
            min_value=min_date, max_value=max_date
        )
    with col2:
        min_val = float(df_raw[target_col].min())
        max_val = float(df_raw[target_col].max())
        val_range = st.slider(
            f"Filter by {target_col} Range:",
            min_value=min_val, max_value=max_val,
            value=(min_val, max_val)
        )

    mask = (
        (df_raw.index.date >= start_date) &
        (df_raw.index.date <= end_date) &
        (df_raw[target_col] >= val_range[0]) &
        (df_raw[target_col] <= val_range[1])
    )
    filtered_df = df_raw.loc[mask]

    st.success(f"Showing {len(filtered_df):,} of {len(df_raw):,} rows.")
    st.dataframe(filtered_df.head(100), use_container_width=True)

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
# TAB 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📈 Time Series Forecasting")
    st.write("Uses the **entire dataset** resampled to the chosen frequency for maximum accuracy.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Configuration")

        model_choice = st.selectbox(
            "Forecasting Model:",
            ["ARIMA", "SARIMA"],
            help="ARIMA: non-seasonal. SARIMA: captures periodic seasonality."
        )

        freq_choice = st.selectbox(
            "Data Frequency (resampling):",
            ["Daily", "Weekly", "Monthly", "Hourly (original)"],
            index=0,
            help=(
                "Daily is recommended — aggregates 145k hourly rows into ~6k daily "
                "averages for much better signal-to-noise and faster fitting."
            )
        )

        # order_mode = st.radio(
        #     "Order Selection:",
        #     ["🤖 Auto (recommended)", "✏️ Manual"],
        #     help="Auto uses ADF stationarity test + ACF/PACF + BIC grid search."
        # )
        order_mode = "🤖 Auto (recommended)"  # Always auto

        # if order_mode == "✏️ Manual":
        #     st.markdown("**ARIMA (p, d, q)**")
        #     p_man = st.number_input("p", 0, 5, 1)
        #     d_man = st.number_input("d", 0, 2, 1)
        #     q_man = st.number_input("q", 0, 5, 1)

        if model_choice == "SARIMA":
            st.markdown("**Seasonal Orders (P, D, Q, s)**")
            # if order_mode == "✏️ Manual":
            #     P_man = st.number_input("P", 0, 3, 1)
            #     D_man = st.number_input("D", 0, 1, 1)
            #     Q_man = st.number_input("Q", 0, 3, 1)
            freq_to_s = {"Daily": 7, "Weekly": 52, "Monthly": 12, "Hourly (original)": 24}
            s_default = freq_to_s[freq_choice]
            s_man = st.number_input(
                "s (season period)",
                min_value=2, max_value=365, value=s_default,
                help=f"Auto-set to {s_default} for {freq_choice} data."
            )

        train_pct = st.slider("Training Data (%)", 50, 95, 80)
        run_btn   = st.button("▶ Run Forecast", type="primary")

    with col2:
        if run_btn:
            # ── Prepare series ──────────────────────────────────────────────
            full_series = df_raw[target_col].dropna()
            resampled   = resample_data(
                full_series.values,
                full_series.index.astype(str).tolist(),
                freq_choice
            )
            resampled   = resampled.dropna()

            st.info(
                f"📦 Using **{len(resampled):,} {freq_choice.lower()} data points** "
                f"(aggregated from {len(full_series):,} hourly rows)."
            )

            split_idx  = int(len(resampled) * (train_pct / 100))
            train_data = resampled.iloc[:split_idx]
            test_data  = resampled.iloc[split_idx:]

            # ── ADF stationarity info ───────────────────────────────────────
            adf_p = adf_test(resampled)
            stat_label = "✅ Stationary" if adf_p < 0.05 else "⚠️ Non-stationary"
            st.caption(f"ADF test p-value: **{adf_p:.4f}** — {stat_label}")

            # ── Order selection ─────────────────────────────────────────────
            if order_mode == "🤖 Auto (recommended)":
                with st.spinner("🔍 Auto-selecting ARIMA orders via BIC grid search..."):
                    best_p, best_d, best_q = suggest_arima_orders(train_data)
                st.success(f"Auto-selected orders: **p={best_p}, d={best_d}, q={best_q}**")
                p, d, q = best_p, best_d, best_q
                if model_choice == "SARIMA":
                    P, D, Q = 1, 1, 1   # sensible SARIMA seasonal defaults
            else:
                p, d, q = p_man, d_man, q_man
                if model_choice == "SARIMA":
                    P, D, Q = P_man, D_man, Q_man

            s = s_man if model_choice == "SARIMA" else None

            # ── ACF / PACF diagnostic plots ─────────────────────────────────
            with st.expander("📊 ACF / PACF Diagnostic Plots"):
                diff_series = train_data.diff(d).dropna() if d > 0 else train_data
                fig_diag, axes = plt.subplots(1, 2, figsize=(13, 3))
                plot_acf(diff_series,  lags=40, ax=axes[0], title="ACF  (differenced series)")
                plot_pacf(diff_series, lags=40, ax=axes[1], title="PACF (differenced series)", method='ywm')
                plt.tight_layout()
                st.pyplot(fig_diag)

            # ── Fit model ───────────────────────────────────────────────────
            with st.spinner(f"Fitting {model_choice}({p},{d},{q})" +
                            (f"×({P},{D},{Q},{s})" if model_choice == "SARIMA" else "") + "…"):
                try:
                    if model_choice == "ARIMA":
                        fitted = ARIMA(train_data, order=(p, d, q)).fit()
                    else:
                        fitted = SARIMAX(
                            train_data,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, s),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        ).fit(disp=False)

                    # ── Forecast ────────────────────────────────────────────
                    fc_obj      = fitted.get_forecast(steps=len(test_data))
                    predictions = fc_obj.predicted_mean
                    conf_int    = fc_obj.conf_int(alpha=0.05)
                    predictions.index = test_data.index
                    conf_int.index    = test_data.index

                    mae, rmse, mape = calculate_metrics(test_data.values, predictions.values)

                    # ── Model summary ───────────────────────────────────────
                    with st.expander(f"📋 {model_choice} Model Summary"):
                        st.text(str(fitted.summary()))

                    # ── Metrics ─────────────────────────────────────────────
                    st.subheader("📊 Model Performance Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE",  f"{mae:.2f} MW")
                    m2.metric("RMSE", f"{rmse:.2f} MW")
                    m3.metric("MAPE", f"{mape:.2f} %")

                    # ── Forecast plot ───────────────────────────────────────
                    fig, ax = plt.subplots(figsize=(14, 6))
                    plot_train = train_data.tail(200)
                    ax.plot(plot_train.index, plot_train,
                            label='Training Data (last 200 pts)', color='steelblue', linewidth=1)
                    ax.plot(test_data.index, test_data,
                            label='Actual', color='green', linewidth=1.2)
                    ax.plot(test_data.index, predictions,
                            label=f'{model_choice} Forecast',
                            color='red' if model_choice == "ARIMA" else 'orange',
                            linestyle='--', linewidth=1.4)
                    ax.fill_between(
                        test_data.index,
                        conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        alpha=0.18,
                        color='red' if model_choice == "ARIMA" else 'orange',
                        label='95% Confidence Interval'
                    )
                    ax.set_title(
                        f"{model_choice} Forecast — {freq_choice} Aggregation\n"
                        f"Orders: ({p},{d},{q})" +
                        (f" × ({P},{D},{Q},{s})" if model_choice == "SARIMA" else ""),
                        fontsize=13
                    )
                    ax.set_ylabel("Consumption (MW)")
                    ax.set_xlabel("Date")
                    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
                    fig.autofmt_xdate()
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # ── Residual plot ───────────────────────────────────────
                    with st.expander("📉 Residual Analysis"):
                        residuals = test_data.values - predictions.values
                        fig_r, axes_r = plt.subplots(1, 2, figsize=(13, 3))
                        axes_r[0].plot(test_data.index, residuals, color='gray', linewidth=0.8)
                        axes_r[0].axhline(0, color='red', linestyle='--')
                        axes_r[0].set_title("Residuals over time")
                        axes_r[0].set_ylabel("Error (MW)")
                        axes_r[0].grid(True, alpha=0.3)
                        axes_r[1].hist(residuals, bins=30, color='steelblue', edgecolor='white')
                        axes_r[1].set_title("Residual Distribution")
                        axes_r[1].set_xlabel("Error (MW)")
                        plt.tight_layout()
                        st.pyplot(fig_r)

                    # ── Forecast table ──────────────────────────────────────
                    st.subheader("🔎 Forecast vs Actual")
                    result_df = pd.DataFrame({
                        "Actual (MW)":   test_data.values,
                        "Forecast (MW)": predictions.values.round(2),
                        "Lower CI":      conf_int.iloc[:, 0].values.round(2),
                        "Upper CI":      conf_int.iloc[:, 1].values.round(2),
                        "Error (MW)":    (test_data.values - predictions.values).round(2)
                    }, index=test_data.index)
                    st.dataframe(result_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Model fitting failed: {e}")
                    st.write("Try switching to Auto order selection, or change the resampling frequency.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔍 Anomaly Detection")
    st.write("Detects abnormal usage patterns using Isolation Forest on the **entire dataset**.")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Parameters")
        contamination = st.slider(
            "Expected Anomaly Rate", 0.01, 0.10, 0.05, step=0.01,
            help="Fraction of data points expected to be anomalies."
        )
        use_features = st.multiselect(
            "Extra Features for Detection:",
            [c for c in numeric_cols if c != target_col],
            default=[],
            help="Adding engineered features (hour, lag, rolling mean) improves detection."
        )
        run_anomaly = st.button("▶ Run Anomaly Detection", type="primary")

    with col2:
        if run_anomaly:
            with st.spinner("Running Isolation Forest on full dataset…"):
                feature_cols = [target_col] + use_features
                data_ad = df_raw[feature_cols].dropna().copy()

                iso = IsolationForest(
                    contamination=contamination,
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
                data_ad['Anomaly'] = iso.fit_predict(data_ad[feature_cols])
                data_ad['Score']   = iso.decision_function(data_ad[feature_cols])

                anomalies = data_ad[data_ad['Anomaly'] == -1]

                # ── Plot ────────────────────────────────────────────────────
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(data_ad.index, data_ad[target_col],
                        label='Consumption', color='steelblue', alpha=0.5, linewidth=0.4)
                ax.scatter(anomalies.index, anomalies[target_col],
                           color='red', label=f'Anomaly ({len(anomalies):,})',
                           zorder=5, s=12, alpha=0.75)
                ax.set_title(
                    f"Isolation Forest Anomaly Detection — {len(anomalies):,} anomalies "
                    f"({contamination*100:.0f}% contamination)",
                    fontsize=13
                )
                ax.set_ylabel("Consumption (MW)")
                ax.set_xlabel("Datetime")
                ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
                fig.autofmt_xdate()
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                # ── Stats ───────────────────────────────────────────────────
                st.subheader("📊 Anomaly Statistics")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Total Anomalies",   f"{len(anomalies):,}")
                a2.metric("Anomaly Rate",       f"{len(anomalies)/len(data_ad)*100:.2f}%")
                a3.metric("Min Anomaly (MW)",   f"{anomalies[target_col].min():,.0f}")
                a4.metric("Max Anomaly (MW)",   f"{anomalies[target_col].max():,.0f}")

                # Anomaly score distribution
                with st.expander("📉 Anomaly Score Distribution"):
                    fig_s, ax_s = plt.subplots(figsize=(10, 3))
                    ax_s.hist(data_ad['Score'], bins=60, color='steelblue',
                              edgecolor='white', alpha=0.8, label='Normal')
                    ax_s.hist(anomalies['Score'], bins=30, color='red',
                              edgecolor='white', alpha=0.8, label='Anomaly')
                    ax_s.axvline(0, color='black', linestyle='--', label='Decision boundary')
                    ax_s.set_title("Isolation Forest Decision Scores")
                    ax_s.set_xlabel("Anomaly Score (more negative = more anomalous)")
                    ax_s.legend()
                    ax_s.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_s)

                # ── Timestamps table ────────────────────────────────────────
                st.subheader("🕒 Anomaly Timestamps")
                display_cols = [target_col, 'Score'] + use_features
                st.dataframe(
                    anomalies[display_cols].sort_values('Score').head(200),
                    use_container_width=True
                )
