# system/inspect_db.py
import sqlite3
import pandas as pd


def inspect_database(db_path="system/live_database.db"):
    print(f"\nInspecting Trading Database: {db_path}")
    print("=" * 50)

    try:
        conn = sqlite3.connect(db_path)

        # 1. The Control Panel
        print("\nSTRATEGY CONTROL PANEL (Who is allowed to trade?)")
        df_control = pd.read_sql_query("SELECT * FROM strategy_control", conn)
        if df_control.empty:
            print("   [Empty] No strategies registered yet.")
        else:
            print(df_control.to_string(index=False))

        # 2. Virtual Positions
        print("\nCURRENT VIRTUAL POSITIONS (What do they own?)")
        df_positions = pd.read_sql_query("SELECT * FROM virtual_positions", conn)
        if df_positions.empty:
            print("   [Empty] No positions held.")
        else:
            print(df_positions.to_string(index=False))

        # 3. Target Signals
        print("\nLATEST TARGET SIGNALS (What do they want the Executor to do?)")
        df_signals = pd.read_sql_query("SELECT * FROM target_signals", conn)
        if df_signals.empty:
            print("   [Empty] No signals published.")
        else:
            print(df_signals.to_string(index=False))

        # 4. Trade Log Summary
        print("\nRECENT TRADE LOGS (Last 5 trades)")
        df_logs = pd.read_sql_query(
            "SELECT * FROM virtual_trade_log ORDER BY timestamp DESC LIMIT 5", conn
        )
        if df_logs.empty:
            print("[Empty] No trades executed yet.")
        else:
            print(df_logs.to_string(index=False))

        print("\n" + "=" * 50 + "\n")
        conn.close()

    except sqlite3.OperationalError as e:
        print(f"❌ Database error: {e}")
        print("Tip: Run your strategy runner at least once to build the tables!")


if __name__ == "__main__":
    inspect_database()
