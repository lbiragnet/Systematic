import sqlite3
import datetime
import pandas as pd
from abc import ABC, abstractmethod


class LiveStrategyBase(ABC):
    def __init__(
        self,
        name: str,
        db_path: str = "system/live_database.db",
        default_capital: float = 10000.0,
        start_active: bool = False,
    ):
        self.name = name
        self.db_path = db_path
        self.default_capital = default_capital
        self.start_active = start_active
        # Automatically register the strategy the moment the class is instantiated
        self._register_if_missing()

    def _register_if_missing(self):
        """Auto-registers the strategy in the database if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if this strategy is already in the control panel
        cursor.execute(
            "SELECT 1 FROM strategy_control WHERE strategy_name = ?", (self.name,)
        )
        if cursor.fetchone() is None:
            print(
                f"FIRST TIME SETUP: Auto-registering '{self.name}' into Control Panel."
            )
            print(
                f"   -> Capital: ${self.default_capital} | Active: {self.start_active}"
            )

            # Insert the defaults
            cursor.execute(
                """
                INSERT INTO strategy_control (strategy_name, is_active, allocated_capital) 
                VALUES (?, ?, ?)
            """,
                (self.name, int(self.start_active), self.default_capital),
            )
            conn.commit()

        conn.close()

    def _is_active(self) -> bool:
        """Checks the Control Panel to see if the strategy is allowed to run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT is_active, allocated_capital FROM strategy_control WHERE strategy_name = ?",
            (self.name,),
        )
        result = cursor.fetchone()
        conn.close()

        if result is None:
            print(
                f"⚠️  Strategy {self.name} not found in Control Panel. Defaulting to INACTIVE."
            )
            return False

        self.allocated_capital = result[1]
        return bool(result[0])

    def _get_virtual_positions(self) -> dict:
        """Retrieves what this specific strategy currently holds."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, quantity, entry_price FROM virtual_positions WHERE strategy_name = ?",
            (self.name,),
        )
        rows = cursor.fetchall()
        conn.close()

        # Returns a dict: {'BTC-USD': {'qty': 0.5, 'price': 60000}}
        return {row[0]: {"qty": row[1], "price": row[2]} for row in rows}

    def _publish_signals(self, target_portfolio: dict, current_prices: dict):
        """
        Calculates the required trades to reach the target portfolio,
        logs them for analysis, and publishes to the Signal Bus.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.datetime.now()

        current_positions = self._get_virtual_positions()

        # 1. Publish to Signal Bus (For the Execution Engine)
        cursor.execute(
            "DELETE FROM target_signals WHERE strategy_name = ?", (self.name,)
        )
        for ticker, target_qty in target_portfolio.items():
            price = current_prices.get(ticker, 0.0)
            cursor.execute(
                """
                INSERT INTO target_signals (strategy_name, ticker, target_quantity, signal_price, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (self.name, ticker, target_qty, price, now),
            )

            # 2. Update Virtual Positions & Log Trades (For your Analysis)
            current_qty = current_positions.get(ticker, {}).get("qty", 0.0)
            delta_qty = target_qty - current_qty

            # If a trade actually happened
            if abs(delta_qty) > 0.0001:
                action = "BUY" if delta_qty > 0 else "SELL"

                # Log the trade
                cursor.execute(
                    """
                    INSERT INTO virtual_trade_log (timestamp, strategy_name, ticker, action, quantity, price)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (now, self.name, ticker, action, abs(delta_qty), price),
                )

                # Update holdings
                if target_qty == 0:
                    cursor.execute(
                        "DELETE FROM virtual_positions WHERE strategy_name = ? AND ticker = ?",
                        (self.name, ticker),
                    )
                else:
                    # Simplified average entry price logic for demonstration
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO virtual_positions (strategy_name, ticker, quantity, entry_price, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (self.name, ticker, target_qty, price, now),
                    )

        conn.commit()
        conn.close()
        print(f"[{self.name}] published signals successfully.")

    @abstractmethod
    def calculate_logic(self, current_positions: dict) -> tuple[dict, dict]:
        """
        The actual trading logic. Must return:
        1. target_portfolio: {'BTC-USD': 1.5, 'GLD': 0.0} (Quantities to hold)
        2. current_prices: {'BTC-USD': 65000, 'GLD': 200} (For logging)
        """
        pass

    def run(self):
        """The main method triggered by the Raspberry Pi cron job."""
        print(f"\nWaking up strategy: {self.name} at {datetime.datetime.now()}")

        if not self._is_active():
            print(
                f"[{self.name}] is currently DISABLED in the Control Panel. Going back to sleep."
            )
            return

        current_positions = self._get_virtual_positions()

        try:
            # Run the specific strategy logic
            target_portfolio, current_prices = self.calculate_logic(current_positions)

            # Push results to the database
            self._publish_signals(target_portfolio, current_prices)
        except Exception as e:
            print(f"❌ [{self.name}] encountered a critical error: {e}")
