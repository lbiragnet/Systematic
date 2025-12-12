# -------------------- IMPORTS --------------------

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
from urllib3 import HTTPResponse
from typing import cast
import os
import json
from massive import RESTClient

from config.load_env import load_api_keys
from backup_manager import perform_backup


# -------------------- CONFIG --------------------

# Database file
STOCKS_DB_NAME = "historical_stock_data.db"

# Fetch API keys
load_api_keys()
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")
EOD_API_KEY = os.getenv("EOD_API_KEY")

# Create client
client = RESTClient(api_key=MASSIVE_API_KEY)


def init_db():
    """Initialise database"""
    conn = sqlite3.connect(STOCKS_DB_NAME)
    c = conn.cursor()

    # Tickers Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            last_updated DATETIME
        )
    """)

    # Prices Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker TEXT,
            timestamp DATETIME,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, timestamp)
        )
    """)
    conn.commit()
    conn.close()
    print(f"Database {STOCKS_DB_NAME} initialized.")


# -------------------- FETCH DATA --------------------


def get_aggregate_bars_massiveapi(
    ticker: str,
    start: str,
    end: str,
    multiplier: int = 1,
    timespan: str = "day",
    adjusted: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV price data for a given ticker and return a clean DataFrame."""

    response = cast(
        HTTPResponse,
        client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start,
            to=end,
            adjusted=adjusted,
            raw=True,
        ),
    )

    data = json.loads(response.data)
    results = data.get("results", [])
    if not results:
        raise ValueError("No data returned for this request.")

    df = pd.DataFrame(results)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.set_index("timestamp")

    return df


# -------------------- UPDATE DATABASE --------------------


def update_database_massiveapi(tickers_list):
    """Update the database"""
    conn = sqlite3.connect(STOCKS_DB_NAME)

    for i, ticker in enumerate(tickers_list):
        # 1. Find out when we last updated this specific ticker
        existing_date = pd.read_sql(
            f"SELECT MAX(timestamp) as last_date FROM daily_prices WHERE ticker='{ticker}'",
            conn,
        )
        last_date = existing_date["last_date"].iloc[0]
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        if last_date:
            # If we have data, start from the next day
            # The database stores full timestamps (YYYY-MM-DD 00:00:00)
            last_dt_obj = pd.to_datetime(last_date)
            start_date = (last_dt_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            # Massive API has a 2 year max lookback for free tier
            start_date = "2020-01-01"

        if start_date > yesterday:
            print(f"{ticker} is already up to date (Data available up to {last_date}).")
            continue

        # 2. Fetch new data - use yesterday instead of today
        try:
            new_data = get_aggregate_bars_massiveapi(ticker, start_date, yesterday)

            if not new_data.empty:
                new_data["ticker"] = ticker
                new_data.reset_index(inplace=True)
                new_data.to_sql("daily_prices", conn, if_exists="append", index=False)
                print(f"Success: Added {len(new_data)} rows for {ticker}")
            else:
                print(f"No new data found for {ticker}")

        except Exception as e:
            # Catch "Plan doesn't include this timeframe" or other API errors
            print(f"⚠ Error updating {ticker}: {e}")

        # Respect rate limits
        if i != len(tickers) - 1:
            time.sleep(15)

    conn.close()
    print("Update routine finished.")


# -------------------- RUN THE UPDATE --------------------

if __name__ == "__main__":
    with open("stocks_list.txt") as f:
        tickers = f.read().splitlines()

    # 1. Initialise database
    init_db()

    # 2. Run the update
    update_database_massiveapi(tickers)

    # 3. Trigger backup after update is done
    print("\n --- Starting Backup ---")
    perform_backup(db_name=STOCKS_DB_NAME, max_backups=10)
