"""
Entry point voor initiale training en/of online bijleren.
"""
import argparse
import time
from datetime import datetime
from config import LAST_MODEL_PATH, BEST_MODEL_PATH
import pandas as pd


from stable_baselines3 import PPO
from src.feature_engineer import add_tech_indicators, merge_with_sentiment
from src.trading_env import MixedTradingEnv
from src.data_collector import get_prices
from src.news_providers import get_news
from src.news_analyzer import aggregate_daily_sentiment
from src.online_trainer import (
    prepare_dataset,
    build_env_from_df,
    initial_or_load,
    continue_training,
    evaluate_on_period,
    save_checkpoints,
)
from config import HISTORY_YEARS, VALIDATION_START, VALIDATION_END


def build_full_dataset(ticker: str):
    prices = get_prices(ticker, years=HISTORY_YEARS, interval="1d")

    # Zorg dat index altijd datetime is
    prices.index = pd.to_datetime(prices.index, errors="coerce")

    # Check of er geldige data is
    if prices.empty or prices.index.min() is pd.NaT:
        raise ValueError(f"⚠️ Geen geldige prijsdata gevonden voor {ticker}")

    start_date = prices.index.min().date().isoformat()
    end_date = prices.index.max().date().isoformat()

    news = get_news(ticker, start_date, end_date)
    daily_sent = aggregate_daily_sentiment(news)

    df = prepare_dataset(prices, daily_sent, ticker)
    return df



def run_initial_train(ticker, steps):
    # 📌 Data laden met historische horizon
    df = get_prices(ticker, years=HISTORY_YEARS, interval="1d")

    # 📌 Features toevoegen
    df = add_tech_indicators(df)
    # Als je sentiment merge wil gebruiken, uncomment deze regel:
    # df = merge_with_sentiment(df, daily_sent)

    # 📌 Environment maken
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(MixedTradingEnv(df))


    # 📌 NIEUW MODEL STARTEN
    model = PPO("MlpPolicy", env, verbose=1)

    print(
        f"🚀 Start training PPO met {steps} stappen en {env.observation_space.shape[0]} features..."
    )
    model.learn(total_timesteps=steps)

    # 📌 Model opslaan
    model.save(LAST_MODEL_PATH)
    print(f"✅ Model opgeslagen naar {LAST_MODEL_PATH}")
    return model


def run_online_loop(ticker: str, online_steps: int, interval_minutes: int):
    # 📌 Eerst dataset en environment maken
    df = build_full_dataset(ticker)
    env = build_env_from_df(df)

    # 📌 Laad bestaand model of initialiseer nieuw
    model = initial_or_load(env)

    # 📌 Startscore bepalen
    best_score = evaluate_on_period(model, df, VALIDATION_START, VALIDATION_END)

    while True:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Fetching latest data/news...")
        df = build_full_dataset(ticker)
        env = build_env_from_df(df)
        model = continue_training(model, env, timesteps=online_steps)

        score = evaluate_on_period(model, df, VALIDATION_START, VALIDATION_END)
        print(f"Validation score: {score:.6f} (best {best_score:.6f})")
        is_best = score >= best_score
        if is_best:
            best_score = score
        save_checkpoints(model, is_best=is_best)
        print("Checkpoints saved. Sleeping...")
        time.sleep(interval_minutes * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="Bijv. MSFT, AAPL")
    ap.add_argument("--train-initial", type=int, default=0, help="Initiale trainingsstappen (eenmalig)")
    ap.add_argument("--online", action="store_true", help="Start online bijleer loop")
    ap.add_argument("--online-steps", type=int, default=5000, help="Stappen per online-iteratie")
    ap.add_argument("--interval", type=int, default=60, help="Minuten tussen online iteraties")
    args = ap.parse_args()

    if args.train_initial > 0:
        run_initial_train(args.ticker, args.train_initial)
    if args.online:
        run_online_loop(args.ticker, args.online_steps, args.interval)
    if not args.online and args.train_initial == 0:
        # alleen data+features maken ter controle
        df = build_full_dataset(args.ticker)
        print(df.tail())


if __name__ == "__main__":
    main()
