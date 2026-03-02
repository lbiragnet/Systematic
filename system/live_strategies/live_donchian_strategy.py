import yfinance as yf
from live_strategy_base import LiveStrategyBase


class LiveDonchianBtcusd(LiveStrategyBase):
    def __init__(self):
        super().__init__(name="donchian_BTC")
        self.ticker = "BTC-USD"
        self.lookback = 200

    def calculate_logic(self, current_positions: dict) -> tuple[dict, dict]:
        print(f"[{self.name}] Fetching market data...")

        # 1. Get Data
        df = yf.download(self.ticker, period="1y", interval="1d", progress=False)

        # FIX: Squeeze the DataFrame into a flat Series and force it to be a float
        close_prices = df["Close"].squeeze()
        current_price = float(close_prices.iloc[-1])

        # 2. Calculate Indicators
        upper_channel = float(
            close_prices.rolling(self.lookback).max().shift(1).iloc[-1]
        )
        lower_channel = float(
            close_prices.rolling(self.lookback).min().shift(1).iloc[-1]
        )

        # 3. Determine Signal
        target_qty = 0.0
        currently_held = current_positions.get(self.ticker, {}).get("qty", 0.0)

        if current_price > upper_channel:
            print(f"Breakout: {current_price} > {upper_channel}. Buying.")
            # Calculate how many shares we can afford with our allocated capital
            target_qty = self.allocated_capital / current_price

        elif current_price < lower_channel:
            print(f"Breakdown: {current_price} < {lower_channel}. Selling to 0.")
            target_qty = 0.0

        else:
            print("Inside channel. Holding current position.")
            target_qty = currently_held  # No change

        # Return Target State and Prices
        return {self.ticker: target_qty}, {self.ticker: current_price}
