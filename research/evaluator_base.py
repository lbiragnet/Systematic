import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from typing import List, Tuple, Union, Dict, Any
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from time_series_strategies import *

warnings.simplefilter(action="ignore", category=FutureWarning)


# ============================================================
# 1. DATA LAYER
# ============================================================


class DataLoader:
    @staticmethod
    def fetch(ticker: str, start_date: str = "2000-01-01") -> pd.DataFrame:
        """Uses yfinance to fetch historical stock data."""
        try:
            print(f"📥 Downloading data for {ticker}...")
            df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                raise ValueError("Empty DataFrame returned.")
            return df
        except Exception as e:
            print(f"❌ Failed to fetch data: {e}")
            return pd.DataFrame()


# ============================================================
# 2. BACKTESTING ENGINE
# ============================================================


class BacktestEngine:

    def __init__(self, cost_bps: float = 0.001):
        self.cost_bps = cost_bps

    def run(
        self, data: pd.DataFrame, strategy: Strategy, params: Dict[str, Any]
    ) -> Tuple[float, float, pd.Series]:
        """Core vectorised backtest logic."""
        try:
            signal = strategy.generate_signal(data, params)
        except Exception as e:
            print(f"⚠️ Strategy execution error: {e}")
            return 0.0, 0.0, pd.Series()

        # Shift signal to avoid lookahead (Signal at Close T -> Trade at Open T+1, or Close T+1 returns)
        # Assuming Close-to-Close returns
        market_returns = np.log(data["Close"] / data["Close"].shift(1))
        aligned_signal = signal.shift(1)

        strategy_returns = aligned_signal * market_returns

        # Transaction Costs
        trades = aligned_signal.diff().abs().fillna(0)
        costs = trades * self.cost_bps

        net_returns = (strategy_returns - costs).dropna()
        pf, sharpe = self._calc_metrics(net_returns)

        return pf, sharpe, net_returns

    def _calc_metrics(self, returns: pd.Series) -> Tuple[float, float]:
        """Compute performance metrics."""
        if returns.empty:
            return 0.0, 0.0
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].abs().sum()
        pf = (
            100.0
            if losses == 0 and gains > 0
            else (gains / losses if losses != 0 else 0.0)
        )

        sharpe = 0.0
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        return pf, sharpe


# ============================================================
# 3. EVALUATOR
# ============================================================


class Evaluator(ABC):
    """
    Interface for different testing methodologies.
    """

    def __init__(self, engine: BacktestEngine):
        self.engine = engine

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, strategy: Strategy, params: Any) -> Any:
        pass


class PermutationBootstrappingEvaluator(Evaluator):
    """
    Legacy bootstrapping test - Shuffles data to check for luck (Monte Carlo).
    """

    def evaluate(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        params: Dict[str, Any],
        n_perms: int = 200,
        verbose: bool = True,
    ):
        if verbose:
            print(f"\n--- Running Permutation Test ({n_perms} runs) ---")
        real_pf, _, _ = self.engine.run(data, strategy, params)

        def single_run(seed):
            perm_data = self._get_permutation(data, seed)
            pf, _, _ = self.engine.run(perm_data, strategy, params)
            return pf

        perm_pfs = Parallel(n_jobs=-1)(delayed(single_run)(i) for i in range(n_perms))

        # Calc Stats
        perm_pfs = np.array(perm_pfs)
        p_value = (np.sum(perm_pfs >= real_pf) + 1) / (n_perms + 1)

        self._plot(real_pf, perm_pfs, p_value)
        return p_value

    def _get_permutation(self, data: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Get random shuffling of prices."""
        np.random.seed(seed)
        log_ret = np.log(data[["Open", "High", "Low", "Close"]]).diff().iloc[1:]
        shuffled = log_ret.sample(frac=1).reset_index(drop=True)
        start_price = data.iloc[0]
        new_prices = np.exp(shuffled.cumsum()) * start_price
        new_prices.index = data.index[1:]
        return new_prices

    def _plot(self, real_pf, perm_pfs, p_value):
        """Plot histogram of results."""
        plt.figure(figsize=(10, 6))
        plt.hist(perm_pfs, bins=30, alpha=0.7, color="skyblue", label="Random PFs")
        plt.axvline(
            real_pf,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Real PF ({real_pf:.2f})",
        )
        plt.title(f"Permutation (Bootstrap) Test (P-Value: {p_value:.4f})")
        plt.legend()
        plt.show(block=False)


class WalkForwardEvaluator(Evaluator):
    """
    Walkforward Test: Sliding window with Out-of-Sample testing.
    """

    def evaluate(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        param_grid: List[Dict],
        train_years: int = 4,
        test_months: int = 6,
        lookback_buffer: int = 1000,
        verbose: bool = True,  # don't display text / plots when using as part of block bootstrap test
    ):
        if verbose:
            print(f"\n--- Running Walk-Forward Optimization ---")
        train_size = int(train_years * 252)
        test_size = int(test_months * 21)
        total_len = len(data)

        wf_returns = []
        current_idx = train_size

        viz_train = []
        viz_test = []

        while current_idx < total_len:
            train_end = current_idx
            train_start = train_end - train_size
            test_end = min(current_idx + test_size, total_len)

            # Data Slices
            train_data = data.iloc[train_start:train_end]
            warmup_start = max(0, train_end - lookback_buffer)
            test_data_warmup = data.iloc[warmup_start:test_end]
            test_data_real = data.iloc[train_end:test_end]

            if test_data_real.empty:
                break

            viz_train.append((train_data.index[0], train_data.index[-1]))
            viz_test.append((test_data_real.index[0], test_data_real.index[-1]))

            # 1. Optimise on training subset
            best_sharpe = -np.inf
            best_params = None
            for p in param_grid:
                _, sharpe, _ = self.engine.run(train_data, strategy, p)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = p

            # 2. Test
            _, _, full_ret = self.engine.run(test_data_warmup, strategy, best_params)
            period_ret = full_ret.reindex(test_data_real.index).fillna(0)
            wf_returns.append(period_ret)

            current_idx += test_size

        final_curve = pd.concat(wf_returns) if wf_returns else pd.Series()
        if not final_curve.empty and verbose:
            self._plot_timeline(viz_train, viz_test)
            self._plot_wf_returns(final_curve)

        return final_curve

    def _plot_timeline(self, train, test):
        """
        Creates a Gantt chart showing the sliding windows.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # For each window, plot a bar for Train and a bar for Test
        for i, (train_rng, test_rng) in enumerate(zip(train, test)):
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
        ax.set_yticks(range(len(train)))
        ax.set_yticklabels([f"Window {i + 1}" for i in range(len(train))])
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

    def _plot_wf_returns(self, final_curve):
        plt.figure(figsize=(12, 6))
        final_curve.cumsum().plot(
            color="#00ff00", title="Walk-Forward Cumulative Returns"
        )
        plt.show(block=False)


class BlockBoostrappingEvaluator(Evaluator):
    """
    Block bootstrapping to test WFA robustness.
    """

    def __init__(self, engine: BacktestEngine, wfa_evaluator: WalkForwardEvaluator):
        super().__init__(engine)
        self.wfa = wfa_evaluator  # Composition: Uses WFA internally

    def evaluate(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        param_grid: List[Dict],
        n_runs: int = 5,
        verbose: bool = True,
    ):
        """Main function for running the evaluator."""
        print(f"\n--- Running Block Bootstrapping Test ({n_runs} runs) ---")
        synthetic_curves = []

        # Real Run
        real_curve = self.wfa.evaluate(data, strategy, param_grid, verbose=False)

        # Block boostrapping Runs
        for i in tqdm(range(n_runs), desc="Block Bootstrapping Progress"):
            synth_data = self._get_block_bootstrap(data, seed=i)
            # We suppress plots for the inner loops
            curve = self.wfa.evaluate(synth_data, strategy, param_grid, verbose=False)
            if not curve.empty:
                synthetic_curves.append(curve.cumsum())

        self._plot_spaghetti(
            real_curve.cumsum() if not real_curve.empty else None, synthetic_curves
        )

    def _get_block_bootstrap(
        self, data: pd.DataFrame, block_size: int = 63, seed: int = None
    ) -> pd.DataFrame:
        """
        Get block bootstrap (permutation) to avoid destroying trends. Ensures length matching
        """
        np.random.seed(seed)

        # Calculate Returns
        log_ret = np.log(data[["Open", "High", "Low", "Close"]]).diff().iloc[1:]
        n_samples = len(log_ret)
        # Create Blocks
        n_blocks = int(np.ceil(n_samples / block_size))
        blocks = [
            log_ret.iloc[i * block_size : (i + 1) * block_size] for i in range(n_blocks)
        ]
        # Reconstruct History - sample until we fill the length
        shuffled_blocks = []
        current_length = 0
        while current_length < n_samples:
            # Pick any random block - random.randint is exclusive on high, so len(blocks) is safe
            ridx = np.random.randint(0, len(blocks))
            block = blocks[ridx]
            shuffled_blocks.append(block)
            current_length += len(block)
        # 4. Concatenate and Trim
        synth_ret = pd.concat(shuffled_blocks).reset_index(drop=True).iloc[:n_samples]
        start_price = data.iloc[0]
        # Cumulative product to get price path
        new_prices = np.exp(synth_ret.cumsum()) * start_price
        # Assign original dates
        new_prices.index = data.index[1:]

        return new_prices

    def _plot_spaghetti(self, real_curve: pd.Series, synth_curves: List[pd.Series]):
        """Plot curves of bootstrapped price series vs real curve."""
        plt.figure(figsize=(10, 6))
        for c in synth_curves:
            plt.plot(range(len(c)), c.values, color="grey", alpha=0.5)
        if real_curve is not None:
            plt.plot(
                range(len(real_curve)),
                real_curve.values,
                color="#00ff00",
                linewidth=2,
                label="Real",
            )
        plt.title("Block Bootstrapping Test")
        plt.legend()
        plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================


if __name__ == "__main__":
    plt.style.use("dark_background")

    # 1. Setup Data & Engine
    ticker = "BTC-USD"
    data = DataLoader.fetch(ticker, start_date="2015-01-01")
    engine = BacktestEngine(cost_bps=0.0005)

    if not data.empty:
        # 2. Setup Strategy
        strat = DonchianBreakout()

        # 3. Choose your Evaluator!

        # --- A. The "Old" Permutation Test ---
        evaluator = PermutationBootstrappingEvaluator(engine)
        evaluator.evaluate(data, strat, {"lookback": 400}, n_perms=100)

        # --- B. The Walk-Forward Test ---
        param_grid = [{"lookback": i} for i in range(200, 500, 20)]
        wf_evaluator = WalkForwardEvaluator(engine)
        wf_evaluator.evaluate(data, strat, param_grid)

        # --- C. The Block Bootstrapping Test ---
        block_evaluator = BlockBoostrappingEvaluator(engine, wf_evaluator)
        block_evaluator.evaluate(data, strat, param_grid, n_runs=15)
