from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


DEFAULT_MODEL_DIR = "/Users/thanyabegum/Desktop/Intelligence Internship"
DEFAULT_DATA_PATH = "/Users/thanyabegum/Downloads/Foreign_Exchange_Rates.xls"

MODEL_DIR = Path(os.environ.get("CURRENCY_MODEL_DIR", DEFAULT_MODEL_DIR))
DATA_PATH = Path(os.environ.get("CURRENCY_DATA_PATH", DEFAULT_DATA_PATH))

CURRENCIES = {
    "Australian Dollar": {
        "code": "AUS",
        "column": "AUSTRALIA - AUSTRALIAN DOLLAR/US$",
    },
    "Brazilian Real": {
        "code": "BRL",
        "column": "BRAZIL - REAL/US$",
    },
    "Canadian Dollar": {
        "code": "CAD",
        "column": "CANADA - CANADIAN DOLLAR/US$",
    },
    "Swiss Franc": {
        "code": "CHF",
        "column": "SWITZERLAND - FRANC/US$",
    },
    "Chinese Yuan": {
        "code": "CNY",
        "column": "CHINA - YUAN/US$",
    },
    "Danish Krone": {
        "code": "DKK",
        "column": "DENMARK - DANISH KRONE/US$",
    },
    "Euro": {
        "code": "EU",
        "column": "EURO AREA - EURO/US$",
    },
    "Great Britain Pound": {
        "code": "GBP",
        "column": "UNITED KINGDOM - UNITED KINGDOM POUND/US$",
    },
    "Hong Kong Dollar": {
        "code": "HKD",
        "column": "HONG KONG - HONG KONG DOLLAR/US$",
    },
    "Indian Rupee": {
        "code": "INR",
        "column": "INDIA - INDIAN RUPEE/US$",
    },
    "Japanese Yen": {
        "code": "JPY",
        "column": "JAPAN - YEN/US$",
    },
    "Korean Won": {
        "code": "KRW",
        "column": "KOREA - WON/US$",
    },
    "Sri Lankan Rupee": {
        "code": "LKR",
        "column": "SRI LANKA - SRI LANKAN RUPEE/US$",
    },
    "Mexican Peso": {
        "code": "MXN",
        "column": "MEXICO - MEXICAN PESO/US$",
    },
    "Malaysian Ringgit": {
        "code": "MYR",
        "column": "MALAYSIA - RINGGIT/US$",
    },
    "Norwegian Krone": {
        "code": "NOK",
        "column": "NORWAY - NORWEGIAN KRONE/US$",
    },
    "New Zealand Dollar": {
        "code": "NZD",
        "column": "NEW ZEALAND - NEW ZELAND DOLLAR/US$",
    },
    "Swedish Krona": {
        "code": "SEK",
        "column": "SWEDEN - KRONA/US$",
    },
    "Singapore Dollar": {
        "code": "SGD",
        "column": "SINGAPORE - SINGAPORE DOLLAR/US$",
    },
    "Thai Baht": {
        "code": "THB",
        "column": "THAILAND - BAHT/US$",
    },
    "Taiwan Dollar": {
        "code": "TWD",
        "column": "TAIWAN - NEW TAIWAN DOLLAR/US$",
    },
    "South African Rand": {
        "code": "ZAR",
        "column": "SOUTH AFRICA - RAND/US$",
    },
}


def model_path_for(currency_name: str) -> Path:
    code = CURRENCIES[currency_name]["code"]
    return MODEL_DIR / f"{code}_model.keras"


@st.cache_data(show_spinner="Loading exchange rate history...")
def load_exchange_rates(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Exchange rate data file was not found: {path}")

    data = pd.read_csv(path)
    data = data.drop(columns=["Unnamed: 24", "Unnamed: 0"], errors="ignore")
    data["Time Serie"] = pd.to_datetime(data["Time Serie"], format="%d-%m-%Y")
    data = data.replace("ND", np.nan)
    data = data.dropna()
    data = data.set_index("Time Serie").sort_index()

    for details in CURRENCIES.values():
        column = details["column"]
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna()

    # Mirrors the notebook's Korea-based outlier cleanup before modelling.
    krw_column = "KOREA - WON/US$"
    if krw_column in data.columns:
        q1 = data[krw_column].quantile(0.25)
        q3 = data[krw_column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        data = data[(data[krw_column] > lower) & (data[krw_column] < upper)]

    return data


@st.cache_resource(show_spinner="Loading forecasting model...")
def load_currency_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file was not found: {path}")

    return load_model(path, compile=False)


def make_forecast(currency_name: str, horizon: int) -> pd.DataFrame:
    details = CURRENCIES[currency_name]
    model_path = model_path_for(currency_name)
    data = load_exchange_rates(str(DATA_PATH))
    model = load_currency_model(str(model_path))

    column = details["column"]
    if column not in data.columns:
        raise KeyError(f"Column '{column}' was not found in the exchange rate data.")

    series = data[column].astype(float).dropna()
    lookback = int(model.input_shape[1] or 12)
    if len(series) < lookback:
        raise ValueError(f"Need at least {lookback} historical values for {currency_name}.")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).flatten()
    window = scaled_values[-lookback:].astype(float).tolist()

    predictions = []
    for _ in range(horizon):
        model_input = np.array(window[-lookback:], dtype=float).reshape(1, lookback, 1)
        next_scaled = float(model.predict(model_input, verbose=0)[0][0])
        predictions.append(next_scaled)
        window.append(next_scaled)

    forecast_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    forecast_dates = pd.bdate_range(series.index.max() + pd.offsets.BDay(1), periods=horizon)

    return pd.DataFrame(
        {
            "Date": forecast_dates,
            "Currency": currency_name,
            "Forecast": forecast_values,
        }
    )


def build_chart(history: pd.Series, forecast: pd.DataFrame, currency_name: str):
    history_tail = history.tail(180).reset_index()
    history_tail.columns = ["Date", "Rate"]
    history_tail["Type"] = "Historical"

    forecast_plot = forecast[["Date", "Forecast"]].rename(columns={"Forecast": "Rate"})
    forecast_plot["Type"] = "Forecast"

    plot_data = pd.concat([history_tail, forecast_plot], ignore_index=True)
    fig = px.line(
        plot_data,
        x="Date",
        y="Rate",
        color="Type",
        markers=True,
        title=f"{currency_name} forecast against US dollar",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Exchange rate",
        legend_title_text="",
        hovermode="x unified",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Currency Forecast", layout="wide")

    st.title("Currency Forecast")
    st.caption("Select a currency and forecast horizon to generate future exchange-rate predictions.")

    with st.sidebar:
        st.header("Forecast controls")
        selected_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=7)
        horizon = st.number_input("Forecast horizon in business days", min_value=1, max_value=365, value=30)
        run_forecast = st.button("Generate forecast", type="primary", use_container_width=True)

        st.divider()
        st.write("Model directory")
        st.code(str(MODEL_DIR), language=None)
        st.write("Data file")
        st.code(str(DATA_PATH), language=None)

    try:
        exchange_rates = load_exchange_rates(str(DATA_PATH))
        selected_column = CURRENCIES[selected_currency]["column"]
        history = exchange_rates[selected_column].astype(float).dropna()

        latest_date = history.index.max().date()
        latest_rate = float(history.iloc[-1])
        model_file = model_path_for(selected_currency)

        metric_cols = st.columns(3)
        metric_cols[0].metric("Latest date", latest_date.strftime("%d %b %Y"))
        metric_cols[1].metric("Latest rate", f"{latest_rate:,.4f}")
        metric_cols[2].metric("Model", model_file.name)

        if run_forecast:
            forecast_df = make_forecast(selected_currency, int(horizon))
            chart = build_chart(history, forecast_df, selected_currency)

            st.subheader("Forecast chart")
            st.plotly_chart(chart, use_container_width=True)

            st.subheader("Forecast table")
            display_df = forecast_df.copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            display_df["Forecast"] = display_df["Forecast"].map(lambda value: f"{value:,.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Choose a currency and click Generate forecast.")

    except Exception as exc:
        st.error(str(exc))
        st.stop()


if __name__ == "__main__":
    main()
