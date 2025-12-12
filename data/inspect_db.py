import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Config
DB_NAME = "historical_stock_data.db"


def get_db_summary():
    """Prints a high-level summary of what is currently in the DB."""
    conn = sqlite3.connect(DB_NAME)

    # Check how many rows we have per ticker and the date range
    query = """
    SELECT 
        ticker, 
        COUNT(*) as row_count, 
        MIN(timestamp) as start_date, 
        MAX(timestamp) as end_date 
    FROM daily_prices 
    GROUP BY ticker
    """
    try:
        df = pd.read_sql(query, conn)
        if df.empty:
            print("⚠ The database is empty.")
        else:
            print("\n--- DATABASE SUMMARY ---")
            print(df.to_string(index=False))
            print("-" * 40)
    except Exception as e:
        print(f"Error reading DB: {e}")
    finally:
        conn.close()


def view_ticker_data(ticker, limit=5):
    """Prints the first N rows for a specific ticker."""
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT * FROM daily_prices WHERE ticker='{ticker}' ORDER BY timestamp ASC LIMIT {limit}"

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"\n--- DATA SAMPLE: {ticker} ---")
    print(df)
    print("-" * 40)


def plot_ticker(ticker):
    """Visualizes the closing price to ensure data looks 'sane'."""
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT timestamp, close FROM daily_prices WHERE ticker='{ticker}' ORDER BY timestamp ASC"

    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print(f"No data found for {ticker} to plot.")
        return

    # Convert timestamp string to actual datetime objects
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["close"], label=ticker, linewidth=1)
    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Show summary of all tickers
    get_db_summary()

    # 2. View raw data sample for one ticker
    view_ticker_data("IBM", limit=50)

    # 3. Plot chart (uncomment to run)
    # plot_ticker("AAPL")
