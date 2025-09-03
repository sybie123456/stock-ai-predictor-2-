
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st

from dashboard_utils import (
    find_best_model, ta_features, data_health_report, choose_ticker_from_models, load_local_news
)

st.set_page_config(page_title="AI Herbegin – Online Dashboard", layout="wide")
st.title("📊 AI Herbegin – Online Dashboard")

DATA_DIR = os.getenv("DATA_DIR", "data_cache")
MODELS_DIR = os.getenv("MODELS_DIR", "models")

with st.sidebar:
    st.header("⚙️ Instellingen")
    # Ticker selectie
    tickers_from_models = choose_ticker_from_models(MODELS_DIR)
    default_ticker = tickers_from_models[0] if tickers_from_models else "AAPL"
    ticker = st.text_input("Ticker", value=default_ticker).upper().strip()
    # Datumrange
    today = datetime.utcnow().date()
    start = st.date_input("Startdatum", value=today - timedelta(days=365))
    end = st.date_input("Einddatum", value=today)
    use_rl = st.checkbox("Laad RL-model (optioneel)", value=False)
    st.caption("Tip: Zet uit als Stable-Baselines niet geïnstalleerd is.")

# ====== Data ophalen ======
@st.cache_data(show_spinner=True)
def fetch_prices(ticker: str, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        df.index.name = "date"
        return df
    except Exception:
        return pd.DataFrame()

prices = fetch_prices(ticker, start, end)
if prices.empty:
    st.error("Kon geen prijsdata ophalen. Controleer je ticker of datumrange.")
    st.stop()

st.subheader(f"📈 {ticker} koers")
fig = go.Figure(data=go.Candlestick(
    x=prices.index, open=prices["open"], high=prices["high"], low=prices["low"], close=prices["close"],
    name="OHLC"
))
fig.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
st.plotly_chart(fig, width="stretch")

# ====== Features & datakwaliteit ======
\1
    # Fix Arrow compatibility: convert object cols to string, dates to datetime
    for c in feat.select_dtypes(include="object").columns:
        feat[c] = feat[c].astype(str)
    if "date" in feat.columns:
        feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
st.subheader("🧪 Datakwaliteit")
st.dataframe(data_health_report(feat), width="stretch")

st.subheader("🧮 Indicatoren (voorbeeld)")
st.dataframe(feat.tail(20), width="stretch")

# ====== Sentiment uit lokale nieuwsbestanden (optioneel) ======
news_df = load_local_news(DATA_DIR)
if not news_df.empty:
    st.subheader("📰 Lokaal nieuws & sentiment")
    analyzer = SentimentIntensityAnalyzer()
    news_df = news_df.copy()
    def _score(text):
        try:
            s = analyzer.polarity_scores(str(text or ""))
            return s["compound"]
        except Exception:
            return 0.0
    news_df["sentiment"] = news_df["title"].astype(str).apply(_score)
    st.dataframe(news_df.tail(50), width="stretch")
else:
    st.info("Geen lokale nieuwsbestanden gevonden in 'data_cache/'.")

# ====== (Optioneel) RL-model laden ======
rl_loaded = False
rl_reason = ""
model_path = find_best_model(MODELS_DIR) if use_rl else None
if model_path:
    try:
        from stable_baselines3 import PPO  # type: ignore
        rl_loaded = True
        rl_reason = f"Model geladen: `{os.path.basename(model_path)}`"
    except Exception as e:
        rl_loaded = False
        rl_reason = f"Stable-Baselines3 niet beschikbaar of importfout: {e}"
elif use_rl:
    rl_reason = "Geen RL-model gevonden in de map 'models/'."
if use_rl:
    st.info(rl_reason)


# ====== Model evaluatie / Backtest ======
st.subheader("📊 Model evaluatie & Backtest")

# Simpele signal: EMA crossover (kort vs lang)
if "short_ema" not in locals():
    prices_local = prices.copy()
short_span = st.sidebar.number_input("Korte EMA span", min_value=3, max_value=50, value=10)
long_span = st.sidebar.number_input("Lange EMA span", min_value=10, max_value=200, value=50)
prices_local["ema_short"] = prices_local["close"].ewm(span=short_span, adjust=False).mean()
prices_local["ema_long"] = prices_local["close"].ewm(span=long_span, adjust=False).mean()
prices_local["signal"] = (prices_local["ema_short"] > prices_local["ema_long"]).astype(int)

from dashboard_utils import evaluate_strategy
backtest_df, back_metrics = evaluate_strategy(prices_local, signal_col="signal")
if backtest_df is not None:
    st.markdown("**Backtest metrics**")
    st.write(back_metrics)
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["strategy_value"], name="Strategy value"))
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["buyhold_value"], name="Buy & Hold"))
    fig_bt.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig_bt, width="stretch")
else:
    st.info("Backtest kon niet uitgevoerd worden (weinig data).")

# ====== Online learning (simpel) ======
st.subheader("🤖 Online bijleren (simulatie)")
from src.online_trainer import OnlineTrainer
if "trainer" not in st.session_state:
    st.session_state.trainer = OnlineTrainer(model_path=os.path.join(MODELS_DIR, "online_sim.zip"))
steps = st.number_input("Aantal trainingsstappen (kort)", min_value=10, max_value=5000, value=200, step=10)
simulate = st.checkbox("Simuleer training (veilig) — geen stable-baselines nodig", value=True)
if st.button("Start online/trainingsstap"):
    with st.spinner("Training... even geduld"):
        hist = st.session_state.trainer.step_train(data_df=prices, steps=int(steps), simulate=simulate)
        # store history in session_state
        st.session_state.last_train = hist

if "last_train" in st.session_state and st.session_state.last_train:
    th = st.session_state.last_train
    df_th = pd.DataFrame(th)
    st.markdown("**Training loss (simulatie)**")
    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=df_th["step"], y=df_th["loss"], name="loss"))
    fig_tr.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_tr, width="stretch")


# ====== Eenvoudige horizon-voorspelling (ARIMA, optioneel) ======
st.subheader("🔮 Dummy-voorspelling (ARIMA)")
horizon_days = st.slider("Voorspelhorizon (dagen)", min_value=3, max_value=30, value=10, step=1)

@st.cache_data(show_spinner=False)
def arima_forecast(close_series: pd.Series, steps: int = 10) -> pd.DataFrame:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        s = close_series.dropna()
        if len(s) < 50:
            return pd.DataFrame()
        model = ARIMA(s, order=(1,1,1))
        res = model.fit()
        fc = res.forecast(steps=steps)
        return pd.DataFrame({"date": pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D"),
                             "forecast": fc.values})
    except Exception:
        return pd.DataFrame()

fc_df = arima_forecast(prices["close"], steps=horizon_days)
if not fc_df.empty:
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=prices.index, y=prices["close"], name="close"))
    f2.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast"], name="forecast"))
    f2.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(f2, width="stretch")
else:
    st.caption("ARIMA forecast niet beschikbaar (te weinig data of ontbrekende dependency).")

# ====== Export ======
st.subheader("⤵️ Download")
export_df = feat.copy()
if not fc_df.empty:
    export_df = export_df.join(fc_df.set_index("date"), how="outer")
st.download_button(
    "Download features & voorspellingen (CSV)",
    data=export_df.to_csv().encode("utf-8"),
    file_name=f"features_voorspellingen_{ticker}.csv"
)
