import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Dict


class PortfolioDataLoader:
    """
    Fetches multiple assets and aligns them into a single 'Wide' DataFrame.
    """

    @staticmethod
    def fetch_universe(
        tickers: List[str], start_date: str = "2015-01-01"
    ) -> pd.DataFrame:
        print(f"📥 Fetching Universe: {tickers}...")

        raw_data = yf.download(
            tickers, start=start_date, auto_adjust=True, progress=True
        )
        if "Close" in raw_data.columns:
            prices = raw_data["Close"]
        else:
            # Fallback for single ticker or flat structure
            prices = raw_data

        prices = prices.dropna(axis=1, how="all")

        # Forward Fill: If Bitcoin trades on Sunday but Stock doesn't,
        # carry forward Friday's stock price so we don't have gaps.
        prices = prices.ffill()

        # Drop rows where ANY asset is missing (Wait until all assets start trading)
        prices = prices.dropna()

        print(f"✅ Universe Ready: {prices.shape[0]} days x {prices.shape[1]} assets")
        return prices


class PortfolioBacktester:
    """
    Implements a 'Ranking & Rotation' strategy.
    """

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.returns = prices.pct_change()

    def run_momentum_strategy(
        self,
        lookback_days: int = 126,
        top_n: int = 3,
        rebalance_freq: str = "ME",
        regime_signal: pd.Series = None,
    ):
        """
        1. Calculate Momentum (Returns over lookback).
        2. At every rebalance period (e.g., Month End), rank assets.
        3. Buy the Top N assets (Equal Weight).
        Has an optional regime filter system.
        """
        print(
            f"\n --- Simple Momentum Strategy (Top {top_n} assets, Lookback {lookback_days}d) ---"
        )
        if regime_signal is not None:
            print("Regime Filter: ACTIVE")

        momentum = self.prices.pct_change(lookback_days)
        rebalance_dates = self.prices.resample(rebalance_freq).last().index

        portfolio_curve = []
        current_holdings = None

        capital = 1.0
        daily_returns = []

        # Align rebalance dates to the price index
        rebalance_indices = [
            self.prices.index.get_indexer([d], method="ffill")[0]
            for d in rebalance_dates
        ]
        is_rebalance_day = np.zeros(len(self.prices), dtype=bool)
        is_rebalance_day[rebalance_indices] = True

        # Array of asset returns for speed
        asset_ret_values = self.returns.values
        mom_values = momentum.values
        n_assets = self.prices.shape[1]

        # Weights vector (starts empty)
        weights = np.zeros(n_assets)

        # Need to align regime signal to integer index 't'
        is_volatile = np.zeros(len(self.prices), dtype=bool)
        if regime_signal is not None:
            # Reindex to match prices exactly, fill NaNs with 0 (assume calm if unknown)
            aligned_regime = regime_signal.reindex(self.prices.index).fillna(0)
            is_volatile = aligned_regime.values == 1

        for t in range(lookback_days, len(self.prices)):
            # A. Calculate P&L from yesterday's holdings
            day_ret = np.dot(weights, asset_ret_values[t])
            daily_returns.append(day_ret)

            # B. Rebalance Logic
            if is_rebalance_day[t]:

                # Enforce a break if regime is highly volatile
                if regime_signal is not None and is_volatile[t]:
                    weights = np.zeros(n_assets)

                else:
                    # Get Momentum Scores for today (t)
                    scores = mom_values[t]

                    # Rank: Get indices of Top N scores
                    valid_indices = np.where(~np.isnan(scores))[0]

                    if len(valid_indices) >= top_n:
                        # Filter scores
                        valid_scores = scores[valid_indices]
                        # Sort indices by score
                        sorted_valid_idx = valid_indices[np.argsort(valid_scores)]
                        # Pick Top N
                        top_indices = sorted_valid_idx[-top_n:]

                        # C. Set New Weights (Equal Weight)
                        weights = np.zeros(n_assets)
                        weights[top_indices] = 1.0 / top_n
                    else:
                        # Cash fallback if not enough data
                        weights = np.zeros(n_assets)

        # 4. Compile Results
        dates = self.prices.index[lookback_days:]
        equity_curve = pd.Series(daily_returns, index=dates).cumsum()

        return equity_curve, weights  # Returns final weights for inspection


# ==========================================
# MAIN EXECUTION (PORTFOLIO)
# ==========================================

# if __name__ == "__main__":
#     plt.style.use("dark_background")

#     # XLK=Tech, XLE=Energy, XLF=Finance, TLT=Bonds, GLD=Gold, BTC=Bitcoin
#     universe = ["XLK", "XLE", "XLF", "TLT", "GLD", "BTC-USD"]
#     loader = PortfolioDataLoader()
#     prices = loader.fetch_universe(universe, start_date="2016-01-01")

#     if not prices.empty:
#         backtester = PortfolioBacktester(prices)
#         # Run Strategy: "Buy the Top 2 winners based on 6-month momentum"
#         equity, final_weights = backtester.run_momentum_strategy(
#             lookback_days=126,
#             top_n=2,
#             rebalance_freq="ME",  # Rebalance Monthly
#         )

#         # Benchmark (Equal Weight Buy & Hold of the Universe)
#         avg_ret = backtester.returns.mean(axis=1).cumsum()
#         avg_ret = avg_ret[equity.index]  # Align dates

#         plt.figure(figsize=(12, 6))
#         plt.plot(equity, color="#00ff00", label="Racing Car (Momentum)", linewidth=2)
#         plt.plot(
#             avg_ret, color="grey", label="Equal Weight Universe (Benchmark)", alpha=0.7
#         )
#         plt.title("Cross-Sectional Momentum: Rotating into Strength")
#         plt.ylabel("Cumulative Log Return")
#         plt.legend()
#         plt.grid(alpha=0.2)
#         plt.show()


# Import the HMM logic (assuming it's in oo_strategy_evaluator.py or you paste the method here)
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def get_regime_signal(prices: pd.DataFrame, trend_window: int = 200):
    """
    Calculates a 'Smart' Traffic Light:
    Signal = 1 (Cash) ONLY IF:
      1. HMM says 'High Volatility'
      AND
      2. Market is in a Downtrend (Price < 200 SMA)
    """
    # 1. Create Benchmark
    benchmark = prices.mean(axis=1)

    # 2. Fit HMM (The Volatility Sensor)
    returns = 100 * np.log(benchmark / benchmark.shift(1)).dropna()
    print("Training Regime Filter (HMM)...")
    model = MarkovRegression(returns, k_regimes=2, trend="c", switching_variance=True)
    res = model.fit(disp=False)

    # Identify Volatile Regime
    sigma_0 = res.params["sigma2[0]"]
    sigma_1 = res.params["sigma2[1]"]
    idx_volatile = 1 if sigma_1 > sigma_0 else 0
    prob_volatile = res.smoothed_marginal_probabilities[idx_volatile]

    # HMM Signal: 1 if Volatile, 0 if Calm
    hmm_signal = (prob_volatile > 0.5).astype(int)

    # --- THE FIX IS HERE ---
    # reindex() converts to float due to NaNs.
    # fillna(0) fills them, but we must explicitly cast back to .astype(int)
    hmm_signal = hmm_signal.reindex(prices.index).fillna(0).astype(int)

    # 3. Trend Sensor (The Direction Filter)
    ma_trend = benchmark.rolling(window=trend_window).mean()
    is_downtrend = (benchmark < ma_trend).astype(int)

    # 4. Composite Signal
    # Now both sides are integers, so '&' will work perfectly
    final_signal = hmm_signal & is_downtrend

    return final_signal


# if __name__ == "__main__":
#     # ... (Setup code same as before) ...
#     plt.style.use("dark_background")
#     # XLK=Tech, XLE=Energy, XLF=Finance, TLT=Bonds, GLD=Gold, BTC=Bitcoin
#     universe = ["XLK", "XLE", "XLF", "TLT", "GLD", "BTC-USD"]
#     loader = PortfolioDataLoader()
#     prices = loader.fetch_universe(universe, start_date="2016-01-01")

#     if not prices.empty:
#         backtester = PortfolioBacktester(prices)

#         # 4. Generate Traffic Light Signal
#         regime_signal = get_regime_signal(prices)

#         # 5. Run Unfiltered Strategy (Benchmark for comparison)
#         equity_raw, _ = backtester.run_momentum_strategy(
#             lookback_days=126, top_n=2, rebalance_freq="M", regime_signal=None
#         )

#         # 6. Run Filtered Strategy (The Smart Car)
#         equity_smart, _ = backtester.run_momentum_strategy(
#             lookback_days=126, top_n=2, rebalance_freq="M", regime_signal=regime_signal
#         )

#         # 7. Plot Comparison
#         plt.figure(figsize=(12, 6))
#         plt.plot(equity_raw, color="grey", label="Raw Momentum (No Brakes)", alpha=0.6)
#         plt.plot(
#             equity_smart,
#             color="#00ff00",
#             label="Smart Momentum (With Regime Filter)",
#             linewidth=2,
#         )

#         # Plot the Regime (Red background for crash mode)
#         # We scale it to fit the chart for visualization
#         ymin, ymax = plt.ylim()
#         plt.fill_between(
#             regime_signal.index,
#             ymin,
#             ymax,
#             where=(regime_signal == 1),
#             color="red",
#             alpha=0.1,
#             label="Crash Regime (Cash)",
#         )

#         plt.title("The Effect of a Regime Filter (Traffic Light)")
#         plt.ylabel("Cumulative Log Return")
#         plt.legend()
#         plt.show()


if __name__ == "__main__":
    # ... Setup ...
    plt.style.use("dark_background")

    universe = ["XLK", "XLE", "XLF", "TLT", "GLD", "BTC-USD"]
    loader = PortfolioDataLoader()
    prices = loader.fetch_universe(universe, start_date="2016-01-01")

    if not prices.empty:

        backtester = PortfolioBacktester(prices)
        # Basic momentum - no synthetic cash
        basic_equity, basic_final_weights = backtester.run_momentum_strategy(
            lookback_days=126,
            top_n=2,
            rebalance_freq="ME",  # Rebalance Monthly
        )

        # Benchmark (Equal Weight Buy & Hold of the Universe)
        basic_avg_ret = backtester.returns.mean(axis=1).cumsum()
        basic_avg_ret = basic_avg_ret[basic_equity.index]  # Align dates

        # --- THE FIX: ADD SYNTHETIC CASH ---
        # Create a column of 1.0s (or a slow upward drift for interest rates)
        # For simplicity, we assume Risk-Free Rate is 0% (Flat line)
        # Ideally, fetch 'SHV' (Short Treasury ETF) instead, but this works for simulation.
        prices["CASH"] = 1.0

        # NOTE: We must ensure CASH doesn't break pct_change (0/0).
        # So we give it a tiny drift or just use a stable ETF like 'SHV' in the real universe.
        # Better yet: Let's fetch 'SHV' (Short-term Treasuries) as the cash proxy.

        # REDO UNIVERSE WITH CASH PROXY
        universe_with_cash = ["XLK", "XLE", "XLF", "TLT", "GLD", "BTC-USD", "SHV"]
        prices = loader.fetch_universe(universe_with_cash, start_date="2016-01-01")

        backtester = PortfolioBacktester(prices)

        # Run Strategy without Regime Signal
        # The strategy can now pick 'SHV' if it ranks in the Top 2
        equity_natural_cash, weights = backtester.run_momentum_strategy(
            lookback_days=126, top_n=2, rebalance_freq="ME", regime_signal=None
        )

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(
            equity_natural_cash,
            color="#00ff00",
            label="Momentum with Natural Cash (SHV)",
            linewidth=1,
        )
        plt.plot(basic_equity, color="cyan", label="Simple Momentum", linewidth=1)
        plt.plot(
            basic_avg_ret,
            color="grey",
            label="Equal Weight Universe (Benchmark)",
            alpha=0.7,
            linewidth=1,
        )
        plt.title(
            "Comparison of Momentum with 'SHV' as a Safety Valve, Simple Momentum, and Benchmark"
        )
        plt.show()

        # Check if it picked Cash in 2022
        # We can look at the 'weights' dataframe or just inspect the curve.
