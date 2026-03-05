# system/toggle_strategy.py
import sqlite3
import argparse


def toggle_strategy(db_path: str, strategy_name: str, state: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Convert 'on'/'off' to 1 or 0
    is_active = 1 if state.lower() == "on" else 0

    cursor.execute(
        """
        UPDATE strategy_control 
        SET is_active = ? 
        WHERE strategy_name = ?
    """,
        (is_active, strategy_name),
    )

    if cursor.rowcount == 0:
        print(f"❌ Error: Strategy '{strategy_name}' not found in the database.")
        print(
            "Tip: Run the strategy once so it auto-registers before trying to toggle it."
        )
    else:
        print(
            f"✅ Strategy '{strategy_name}' is now {'🟢 ON' if is_active else '🔴 OFF'}."
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a strategy ON or OFF.")
    parser.add_argument(
        "strategy_name",
        type=str,
        help="Exact name in the database (e.g., donchian_BTC)",
    )
    parser.add_argument(
        "state", type=str, choices=["on", "off"], help="Turn the strategy 'on' or 'off'"
    )

    args = parser.parse_args()

    # Note: Ensure this path matches your actual DB location!
    toggle_strategy("system/live_database.db", args.strategy_name, args.state)
