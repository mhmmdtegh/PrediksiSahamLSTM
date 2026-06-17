import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Saham LSTM",
    layout="wide"
)
st.title("📈 Prediksi Harga Saham Menggunakan LSTM")

# INPUT
col1, col2, col3 = st.columns(3)
with col1:
    kode_saham = st.text_input(
        "Kode Saham",
        "BBCA.JK"
    )
with col2:
    tanggal_mulai = st.date_input(
        "Tanggal Mulai",
        pd.to_datetime("2020-01-01")
    )
with col3:
    epoch = st.number_input(
        "Epoch",
        min_value=1,
        max_value=100,
        value=10
    )

# TOMBOL
if st.button("Prediksi"):

    # DOWNLOAD DATA
    with st.spinner("Mengambil data dan melatih LSTM..."):
        data = yf.download(
            kode_saham,
            start=tanggal_mulai,
            auto_adjust=True,
            progress=False
        )
    if data.empty:
        st.error("Data saham tidak ditemukan")
        st.stop()
    # Ambil kolom Close dengan aman
    if isinstance(data["Close"], pd.DataFrame):
        close_prices = data["Close"].iloc[:, 0]
    else:
        close_prices = data["Close"]

    # INFO TERKINI
    harga_terakhir = float(close_prices.iloc[-1])
    st.metric(
        "Harga Penutupan Terakhir",
        f"Rp {harga_terakhir:,.2f}"
    )
    # GRAFIK HISTORIS
    st.subheader("Grafik Harga Historis")
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=data.index,
            y=close_prices,
            mode="lines",
            name="Harga Historis"
        )
    )
    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )
    
    # SCALING
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(
        close_prices.values.reshape(-1, 1)
    )

    # WINDOW 60
    X = []
    y = []
    window_size = 60
    for i in range(window_size, len(scaled_data)):
        X.append(
            scaled_data[i-window_size:i]
        )
        y.append(
            scaled_data[i]
        )
    X = np.array(X)
    y = np.array(y)

    # TRAIN TEST SPLIT
    split = int(len(X) * 0.8)
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    
    # RESHAPE
    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )
    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )
    
    # MODEL LSTM
    st.subheader("Training LSTM")
    model = Sequential()
    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(60, 1)
        )
    )
    model.add(
        LSTM(50)
    )
    model.add(
        Dense(25)
    )
    model.add(
        Dense(1)
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=32,
        verbose=0
    )
    loss_akhir = history.history["loss"][-1]
    st.success("Training selesai")
    st.metric(
        "Loss Terakhir",
        round(loss_akhir, 6)
    )

    # PREDIKSI TEST
    predictions = model.predict(
        X_test,
        verbose=0
    )
    predictions = scaler.inverse_transform(
        predictions
    )
    actual_prices = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )
    
    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            predictions
        )
    )
    st.metric(
        "RMSE",
        round(rmse, 2)
    )

    # DATAFRAME HASIL TEST
    test_dates = close_prices.index[
        len(close_prices) - len(actual_prices):
    ]
    hasil = pd.DataFrame({
        "Tanggal": test_dates,
        "Harga Asli": actual_prices.flatten(),
        "Harga Prediksi": predictions.flatten()
    })
    st.subheader(
        "Harga Asli vs Harga Prediksi"
    )
    st.dataframe(
        hasil.tail(30)
    )

    # GRAFIK TEST
    fig_compare = go.Figure()
    fig_compare.add_trace(
        go.Scatter(
            x=hasil["Tanggal"],
            y=hasil["Harga Asli"],
            mode="lines",
            name="Harga Asli"
        )
    )
    fig_compare.add_trace(
        go.Scatter(
            x=hasil["Tanggal"],
            y=hasil["Harga Prediksi"],
            mode="lines",
            name="Prediksi"
        )
    )
    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )
    
    # FORECAST 30 HARI
    last_60_days = scaled_data[-60:].copy()
    future_predictions = []
    for _ in range(30):
        x_input = np.array(
            last_60_days
        ).reshape(
            1,
            60,
            1
        )
        pred = model.predict(
            x_input,
            verbose=0
        )
        future_predictions.append(
            pred[0][0]
        )
        last_60_days = np.vstack(
            [last_60_days[1:], pred]
        )
    future_predictions = np.array(
        future_predictions
    ).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(
        future_predictions
    )

    st.write("Harga terakhir:", float(close_prices.iloc[-1]))
    st.write(
    "Prediksi hari pertama:",
    float(future_predictions[0][0])
    )
    
    # TANGGAL MASA DEPAN
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=30
    )
    forecast_df = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi Harga": future_predictions.flatten()
    })
    st.subheader(
        "Tabel Prediksi 30 Hari"
    )
    st.dataframe(
        forecast_df
    )

    # GRAFIK FORECAST
    fig_forecast = go.Figure()
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df["Tanggal"],
            y=forecast_df["Prediksi Harga"],
            mode="lines+markers",
            name="Forecast"
        )
    )
    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )
    
    # GRAFIK HISTORIS + FORECAST
    st.subheader("Historis + Forecast 30 Hari")

    # Forecast hanya untuk tanggal masa depan (jangan sambungkan ke historis)
    forecast_plot_df = pd.DataFrame({
        "Tanggal": future_dates,
        "Prediksi Harga": future_predictions.flatten()
    })

    fig_final = go.Figure()

    # Data historis (garis solid)
    fig_final.add_trace(
        go.Scatter(
            x=data.index,
            y=close_prices,
            mode="lines",
            name="Historis",
            line=dict(dash="solid", color="#1f77b4", width=2)
        )
    )

    # Tandai titik terakhir historis sebagai marker
    fig_final.add_trace(
        go.Scatter(
            x=[data.index[-1]],
            y=[float(close_prices.iloc[-1])],
            mode="markers",
            name="Titik Terakhir Historis",
            marker=dict(size=8, color="#1f77b4")
        )
    )

    # Forecast (sama tipe garis dengan historis)
    fig_final.add_trace(
        go.Scatter(
            x=forecast_plot_df["Tanggal"],
            y=forecast_plot_df["Prediksi Harga"],
            mode="lines",
            name="Forecast 30 Hari",
            line=dict(dash="solid", color="#FF7F0E", width=2)
        )
    )

    st.plotly_chart(
        fig_final,
        use_container_width=True
    )

    harga_prediksi_akhir = float(
        
        future_predictions[-1][0]
    )

    perubahan = (
        (harga_prediksi_akhir - harga_terakhir)
        / harga_terakhir
    ) * 100

    st.metric(
        "Potensi Perubahan 30 Hari",
        f"{perubahan:.2f}%"
    )

    # TOMBOL KELUAR
    if st.button("Keluar"):
        st.write("Sesi dihentikan. Terima kasih.")
        st.stop()