import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from typing import List, Callable, Tuple, Union
from joblib import Parallel, delayed
import warnings

# Suppress pandas fragmentation warnings for heavy backtests
warnings.simplefilter(action="ignore", category=FutureWarning)

# ---------------------------- 1. DATA INGESTION ---------------------------- #


def fetch_historical_data(
    ticker: str, start_date: str = "2000-01-01", end_date: str = None
) -> pd.DataFrame:
    """
    Fetch historical price data using yfinance.
    Uses 'auto_adjust=True' to handle splits and dividends.
    """
    try:
        print(f"Downloading data for {ticker}...")
        # auto_adjust=True fixes OHLC for splits/dividends (Crucial for backtesting)
        df = yf.download(
            ticker, start=start_date, end=end_date, auto_adjust=True, progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise ValueError("Empty DataFrame returned.")

        return df
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return pd.DataFrame()


# ---------------------------- 2. STRATEGY DEFINITIONS ---------------------------- #


def moving_average_crossover(returns: pd.DataFrame, short: int, long: int) -> pd.Series:
    """Vectorized Moving Average Crossover."""
    close = returns["Close"]
    ma_short = close.rolling(window=short).mean()
    ma_long = close.rolling(window=long).mean()
    signal = pd.Series(0, index=returns.index)
    signal[ma_short > ma_long] = 1
    signal[ma_short < ma_long] = -1
    return signal


def donchian_breakout(returns: pd.DataFrame, lookback: int) -> pd.Series:
    """Vectorized Donchian Channel Breakout."""
    close = returns["Close"]
    upper = close.rolling(window=lookback).max().shift(1)  # Shift 1 to avoid lookahead
    lower = close.rolling(window=lookback).min().shift(1)
    signal = pd.Series(0, index=returns.index)
    # Buy if we break above previous high, Sell if we break below previous low
    signal[close > upper] = 1
    signal[close < lower] = -1
    return signal.ffill().fillna(0)  # Trend following: hold until flip


def mean_reversion_zscore(
    returns: pd.DataFrame, lookback: int, entry_z: float = 2.0
) -> pd.Series:
    """Mean Reversion based on Z-Score from Moving Average."""
    close = returns["Close"]
    ma = close.rolling(window=lookback).mean()
    std = close.rolling(window=lookback).std()
    z_score = (close - ma) / std
    signal = pd.Series(0, index=returns.index)
    signal[z_score > entry_z] = -1  # Sell when expensive
    signal[z_score < -entry_z] = 1  # Buy when cheap
    return signal


# ---------------------------- 3. CORE BACKTEST ENGINE ---------------------------- #


def evaluate_performance(strat_returns: pd.Series) -> Tuple[float, float, float]:
    """Calculates Profit Factor, Sharpe, and Total Return."""
    if strat_returns.empty:
        return 0.0, 0.0, 0.0

    gains = strat_returns[strat_returns > 0].sum()
    losses = strat_returns[strat_returns < 0].abs().sum()

    if losses == 0:
        pf = 100.0 if gains > 0 else 0.0
    else:
        pf = gains / losses

    sharpe = 0.0
    if strat_returns.std() > 0:
        sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252)

    total_ret = strat_returns.sum()
    return pf, sharpe, total_ret


def run_backtest(
    returns: pd.DataFrame, strategy: Callable, params: dict, cost_bps: float = 0.001
) -> Tuple[float, float, pd.Series]:
    """
    Run a vectorized backtest with Transaction Costs.

    Args:
        returns: OHLC DataFrame
        strategy: Function returning a signal Series (-1, 0, 1)
        params: Dict of parameters for the strategy
        cost_bps: Cost per trade in decimal (0.001 = 10 basis points = 0.1%)
    """
    # 1. Generate Signal
    # We pass 'returns' (OHLC) to the strategy
    # Important: The strategy must handle its own parameter casting
    try:
        signal = strategy(returns, **params)
    except TypeError as e:
        # Handle cases where param names don't match exactly
        print(f"⚠️ Parameter mismatch for {strategy.__name__}: {e}")
        return 0.0, 0.0, pd.Series()

    # 2. Shift Signal to avoid Look-Ahead Bias
    # Signal generated at Close of Day T affects returns of Day T+1
    # We use 'Close-to-Close' returns
    market_returns = np.log(returns["Close"] / returns["Close"].shift(1))

    # Align signal: Signal[t] trades on Market_Returns[t+1]
    # We .shift(1) the signal so it aligns with the FUTURE return
    aligned_signal = signal.shift(1)

    # 3. Calculate Gross Returns
    strategy_returns = aligned_signal * market_returns

    # 4. Calculate Transaction Costs
    # We pay costs whenever the signal CHANGES (Diff != 0)
    trades = aligned_signal.diff().abs().fillna(0)
    costs = trades * cost_bps

    # 5. Net Returns
    net_returns = strategy_returns - costs
    net_returns = net_returns.dropna()

    pf, sharpe, _ = evaluate_performance(net_returns)
    return pf, sharpe, net_returns


# ---------------------------- 4. PERMUTATION LOGIC ---------------------------- #


def get_permutation(
    returns: pd.DataFrame, start_index: int = 0, seed=None
) -> pd.DataFrame:
    """
    Generates a synthetic price series preserving volatility and drift characteristics.
    """
    np.random.seed(seed)

    # Calculate log returns
    log_ret = np.log(returns[["Open", "High", "Low", "Close"]]).diff().iloc[1:]

    # Shuffle the returns
    # We shuffle the time axis (rows)
    shuffled_ret = log_ret.sample(frac=1).reset_index(drop=True)

    # Reconstruct Price Paths
    # Start from the first real price
    start_price = returns.iloc[0]

    # Cumulative sum of shuffled log returns
    cum_ret = shuffled_ret.cumsum()

    # Project new prices
    # Price_t = Start * exp(CumSum_t)
    new_prices = np.exp(cum_ret) * start_price

    # Assign the original dates index to the new data (to keep pandas happy)
    # We trim the original index to match length (since we lost 1 row due to diff)
    new_prices.index = returns.index[1:]

    return new_prices


def permutation_test(
    returns: pd.DataFrame,
    strategy: Callable,
    params: dict,
    n_permutations: int = 200,
    n_jobs: int = -1,
):
    """
    Compares Real Strategy Performance vs Random Data Performance.
    """
    print(f"\n--- Running Permutation Test ({n_permutations} runs) ---")

    # 1. Real Performance
    real_pf, _, _ = run_backtest(returns, strategy, params)
    print(f"Real PF: {real_pf:.2f}")

    # 2. Parallel Randomized Performance
    def single_run(seed):
        perm_data = get_permutation(returns, seed=seed)
        pf, _, _ = run_backtest(perm_data, strategy, params)
        return pf

    perm_pfs = Parallel(n_jobs=n_jobs)(
        delayed(single_run)(seed) for seed in range(n_permutations)
    )

    # 3. Stats & Plotting
    perm_pfs = np.array(perm_pfs)
    p_value = (np.sum(perm_pfs >= real_pf) + 1) / (n_permutations + 1)

    plt.figure(figsize=(14, 8))
    plt.hist(perm_pfs, bins=30, alpha=0.7, label="Random Data PFs", color="skyblue")
    plt.axvline(
        real_pf,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Real PF ({real_pf:.2f})",
    )
    plt.title(f"Monte Carlo Permutation Test\nP-Value: {p_value:.4f} (Lower is better)")
    plt.xlabel("Profit Factor")
    plt.legend()
    plt.show(block=False)


def plot_synthetic_paths(returns: pd.DataFrame, n_paths: int = 10):
    """Visualizes N random price paths vs the Real path."""
    plt.figure(figsize=(14, 8))

    # Plot random paths
    for i in range(n_paths):
        perm_data = get_permutation(returns, seed=i)
        # Rebase to 1.0 for comparison
        normalized_perm = perm_data["Close"] / perm_data["Close"].iloc[0]
        plt.plot(
            normalized_perm.index, normalized_perm, color="grey", alpha=0.3, linewidth=1
        )

    # Plot real path
    real_norm = returns["Close"] / returns["Close"].iloc[0]
    plt.plot(
        real_norm.index,
        real_norm,
        color="#00ff00",
        linewidth=2,
        label="Real Market Data",
    )

    plt.title(f"Permutation Test: {n_paths} Synthetic Worlds vs Reality")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.show(block=False)


# ---------------------------- 5. WALK-FORWARD ENGINE ---------------------------- #


def plot_walkforward_timeline(train_periods: List[Tuple], test_periods: List[Tuple]):
    """
    Creates a Gantt chart showing the sliding windows.
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    # For each window, plot a bar for Train and a bar for Test
    for i, (train_rng, test_rng) in enumerate(zip(train_periods, test_periods)):
        # Train Bar (Blue)
        ax.barh(
            y=i,
            width=(train_rng[1] - train_rng[0]).days,
            left=train_rng[0],
            color="#1f77b4",
            edgecolor="black",
            alpha=0.8,
        )
        # Test Bar (Orange)
        ax.barh(
            y=i,
            width=(test_rng[1] - test_rng[0]).days,
            left=test_rng[0],
            color="#ff7f0e",
            edgecolor="black",
            alpha=0.8,
        )

    # Decoration
    ax.set_yticks(range(len(train_periods)))
    ax.set_yticklabels([f"Window {i + 1}" for i in range(len(train_periods))])
    ax.set_xlabel("Date")
    ax.set_title(
        " Walk-Forward Optimization Process: Training (Blue) vs Testing (Orange)"
    )

    # Format X-axis dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    legend_elements = [
        Patch(facecolor="#1f77b4", label="In-Sample (Optimization)"),
        Patch(facecolor="#ff7f0e", label="Out-of-Sample (Trading)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.show(block=False)


def walkforward_optimization(
    returns: pd.DataFrame,
    strategy: Callable,
    param_grid: List[dict],
    train_years: int = 4,
    test_months: int = 3,
    cost_bps: float = 0.001,
    enable_benchmark: bool = True,  # Toggle this to True/False
    benchmark_ticker: str = "SPY",
    lookback_buffer: int = 200,  # Keeps the "Cold Start" fix
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Unified Walk-Forward Optimization Engine.
    - Handles standard WFA.
    - Handles Benchmarking.
    - Handles Lookback/Warmup buffering.
    - Generates timeline visualization.
    """

    # 1. OPTIONAL: Fetch Benchmark
    bench_ret = pd.Series()
    if enable_benchmark:
        bench_df = fetch_historical_data(
            benchmark_ticker,
            start_date=returns.index[0].strftime("%Y-%m-%d"),
            end_date=returns.index[-1].strftime("%Y-%m-%d"),
        )
        if not bench_df.empty:
            bench_ret = np.log(bench_df["Close"] / bench_df["Close"].shift(1))
        else:
            print("Warning: Benchmark data empty. Proceeding without benchmark.")
            enable_benchmark = False

    # 2. SETUP
    train_size = int(train_years * 252)
    test_size = int(test_months * 21)
    total_len = len(returns)

    wf_returns = []
    wf_bench_returns = []

    # Store dates for visualization
    viz_train_dates = []
    viz_test_dates = []

    current_idx = train_size
    print(f"Starting Walk-Forward ({train_years}y train -> {test_months}m test)...")

    # 3. MAIN LOOP
    while current_idx < total_len:
        train_start = current_idx - train_size
        train_end = current_idx
        test_end = min(current_idx + test_size, total_len)

        # Optimization Data
        train_data = returns.iloc[train_start:train_end]

        # Test Data (Real window for results)
        test_data_real = returns.iloc[train_end:test_end]

        # Test Data (With Warmup Buffer for Indicators)
        warmup_start = max(0, train_end - lookback_buffer)
        test_data_with_warmup = returns.iloc[warmup_start:test_end]

        if test_data_real.empty:
            break

        # Store dates for timeline plot
        viz_train_dates.append((train_data.index[0], train_data.index[-1]))
        viz_test_dates.append((test_data_real.index[0], test_data_real.index[-1]))

        # --- A. OPTIMIZATION (In-Sample) ---
        best_sharpe = -np.inf
        best_params = None
        for params in param_grid:
            _, sharpe, _ = run_backtest(train_data, strategy, params, cost_bps)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        # --- B. TEST EXECUTION (Out-of-Sample) ---
        # Run on warmup data to prime indicators
        _, _, full_period_returns = run_backtest(
            test_data_with_warmup, strategy, best_params, cost_bps
        )

        # Slice to keep ONLY the real test window
        period_returns = full_period_returns.reindex(test_data_real.index).fillna(0)
        wf_returns.append(period_returns)

        # --- C. BENCHMARK SLICING ---
        if enable_benchmark:
            period_bench = bench_ret.reindex(test_data_real.index).fillna(0)
            wf_bench_returns.append(period_bench)

        current_idx += test_size

    # 4. VISUALIZATION
    plot_walkforward_timeline(viz_train_dates, viz_test_dates)

    # 5. RETURN
    if not wf_returns:
        return (pd.Series(), pd.Series()) if enable_benchmark else pd.Series()

    final_strat = pd.concat(wf_returns)

    if enable_benchmark:
        final_bench = pd.concat(wf_bench_returns)
        return final_strat, final_bench
    else:
        return final_strat


# ---------------------------- MAIN ---------------------------- #

if __name__ == "__main__":
    plt.style.use("dark_background")
    TICKER = "SPY"
    COSTS = 0.0005  # 5 bps

    data = fetch_historical_data(TICKER, start_date="2005-01-01")

    if not data.empty:
        param_grid = [{"lookback": i} for i in range(20, 150, 10)]

        # --- TEST 1: IN-SAMPLE EXCELLENCE ---
        print("\n=== TEST 1: In-Sample Excellence (Baseline) ===")
        # This checks: "Does a single best parameter set work on the whole history?"
        # It is biased, but if this fails, everything fails.
        best_pf, best_sharpe, _ = run_backtest(
            data, donchian_breakout, {"lookback": 50}, COSTS
        )
        print(f"Baseline Result -> PF: {best_pf:.2f} | Sharpe: {best_sharpe:.2f}")

        # --- TEST 2: PERMUTATION (DATA MINING BIAS) ---
        print("\n=== TEST 2: Permutation Test (Checking for Luck) ===")
        # Visualize the synthetic worlds first
        plot_synthetic_paths(data, n_paths=15)
        # Uncomment below to run the actual statistical test (takes time)
        permutation_test(
            data, donchian_breakout, {"lookback": 50}, n_permutations=10000
        )

        # --- TEST 3: WALK-FORWARD ANALYSIS (ROBUSTNESS) ---
        print("\n=== TEST 3: Walk-Forward Analysis (Realistic Simulation) ===")
        # Returns just one Series
        wf_results = walkforward_optimization(
            data,
            donchian_breakout,
            param_grid,
            enable_benchmark=False,  # <--- Flag
        )

        if not wf_results.empty:
            pf_wf, sharpe_wf, _ = evaluate_performance(wf_results)
            print("\nFinal Walk-Forward Performance:")
            print(f"PF: {pf_wf:.2f} | Sharpe: {sharpe_wf:.2f}")

            plt.figure(figsize=(14, 8))
            wf_results.cumsum().plot(
                color="#00ff00",
                title=f"Cumulative Log Returns (Walk-Forward): {TICKER}",
            )
            plt.ylabel("Log Return")
            plt.grid(alpha=0.2)
            plt.show(block=False)

        # Returns Tuple (Strategy, Benchmark)
        wf_results, wf_bench = walkforward_optimization(
            data,
            donchian_breakout,
            param_grid,
            enable_benchmark=True,  # <--- Flag
        )

        if not wf_results.empty:
            plt.figure(figsize=(14, 8))

            # Calculate Cumulative Returns
            strat_cum = wf_results.cumsum()
            bench_cum = wf_bench.cumsum()

            plt.plot(
                strat_cum.index,
                strat_cum,
                label="Strategy (Walk-Forward)",
                color="#00ff00",
            )
            plt.plot(
                bench_cum.index,
                bench_cum,
                label="S&P 500 (Benchmark)",
                color="orange",
                alpha=0.7,
            )

            plt.title("Walk-Forward Performance vs Benchmark")
            plt.ylabel("Cumulative Log Return")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()
