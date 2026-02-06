import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from typing import List, Dict
from tqdm import tqdm

from evaluator_base import (
    DataLoader,
    Evaluator,
    BacktestEngine,
    WalkForwardEvaluator,
    BlockBoostrappingEvaluator,
)
from time_series_strategies import Strategy, DonchianBreakout


# ============================================================
# REGIME SWITCHING
# ============================================================


class RegimeSwitchingEvaluator(Evaluator):
    """
    Regime switching generation:
    1. Fits a Hidden Markov Model (HMM) to historical returns.
    2. Learns 2 regimes: 'Calm' (Low Vol) and 'Volatile' (High Vol).
    3. Generates N synthetic price paths based on these physics.
    4. Tests if the strategy survives in these 'parallel universes'.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        wfa_evaluator: WalkForwardEvaluator,
        realistic_ohlc: bool = False,
        tail_heaviness: float = 3.0,  # degrees of freedom of student t
    ):
        super().__init__(engine)
        self.wfa = wfa_evaluator
        self.realistic_ohlc = realistic_ohlc
        self.tail_heaviness = tail_heaviness

    def evaluate(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        param_grid: List[Dict],
        n_runs: int = 5,
    ):
        """Main function for running the evaluator."""
        print(f"\n--- Running Regime-Switching Monte Carlo ({n_runs} runs) ---")

        # Fit the Model and generate paths
        print("Training Hidden Markov Model on history...")
        model_params = self._fit_hmm(data)
        synthetic_curves = []
        # Compare with real curve
        real_curve = self.wfa.evaluate(data, strategy, param_grid, verbose=False)

        for i in tqdm(range(n_runs), desc="Simulating Regimes"):
            # Generate a new history of the same length as real data
            synth_data = self._generate_path(
                model_params,
                length=len(data),
                start_price=data.iloc[0]["Close"],
                seed=i,
            )
            curve = self.wfa.evaluate(synth_data, strategy, param_grid, verbose=False)
            if not curve.empty:
                synthetic_curves.append(curve.cumsum())
        print(synth_data.head())
        self._plot_spaghetti(
            real_curve.cumsum() if not real_curve.empty else None, synthetic_curves
        )

    def _fit_hmm(self, data: pd.DataFrame) -> Dict:
        """
        Uses statsmodels to find the Hidden States (Regimes).
        """
        # Prepare returns (multiply by 100 for numerical stability in optimizer)
        returns = 100 * np.log(data["Close"] / data["Close"].shift(1)).dropna()

        # Fit Markov Regression
        # k_regimes=2 (Bull/Bear), switching_variance=True (different risk in each)
        model = MarkovRegression(
            returns, k_regimes=2, trend="c", switching_variance=True
        )
        res = model.fit(disp=False)

        # Extract Parameters
        # Regime 0 is usually 'Low Vol', Regime 1 is 'High Vol' (but we check/sort them)
        params = {
            "p_00": res.params["p[0->0]"],  # Prob of staying in Regime 0
            "p_10": res.params["p[1->0]"],  # Prob of switching 1 -> 0
            "mu_0": res.params["const[0]"] / 100,  # Rescale back to decimal
            "mu_1": res.params["const[1]"] / 100,
            "sigma_0": np.sqrt(res.params["sigma2[0]"]) / 100,
            "sigma_1": np.sqrt(res.params["sigma2[1]"]) / 100,
        }
        return params

    def _generate_path(
        self,
        params: Dict,
        length: int,
        start_price: float,
        seed: int,
        wick_multiplier: float = 0.5,  # Adjusts how "wild" the wicks are relative to daily vol
    ) -> pd.DataFrame:
        """
        Custom Monte Carlo engine using the fitted HMM parameters.
        """
        np.random.seed(seed)

        # Unpack params
        p_00 = params["p_00"]
        p_11 = 1 - params["p_10"]

        mus = [params["mu_0"], params["mu_1"]]
        sigmas = [params["sigma_0"], params["sigma_1"]]

        trans_mat = [[p_00, 1 - p_00], [1 - p_11, p_11]]

        current_state = 0
        synth_rets = []
        synth_states = []  # Keep track of states to know the vol for H/L generation

        # 1. Generate Returns Loop
        for _ in range(length):
            current_state = np.random.choice([0, 1], p=trans_mat[current_state])
            synth_states.append(current_state)

            # Fat tail generation - use standard_t instead of normal
            t_random = np.random.standard_t(df=self.tail_heaviness)
            # Scale by regime volatility
            r = mus[current_state] + (t_random * sigmas[current_state])

            # OLD (GAUSSIAN) Draw return from the current regime
            # r = np.random.normal(mus[current_state], sigmas[current_state])

            synth_rets.append(r)

        # 2. Reconstruct Price Path (The "Close")
        cum_ret = np.array(synth_rets).cumsum()
        close_path = start_price * np.exp(cum_ret)

        # Create Dates
        dates = pd.date_range(start="2050-01-01", periods=length, freq="D")

        # 3. Construct OHLC
        if not self.realistic_ohlc:
            # Simple Mode: O=H=L=C
            return pd.DataFrame(
                {
                    "Open": close_path,
                    "High": close_path,
                    "Low": close_path,
                    "Close": close_path,
                },
                index=dates,
            )

        else:
            # Realistic Mode: Simulate O/H/L based on Open-Close path + Volatility

            # Open is yesterday's Close (shifted); we assume O[0] = start_price
            open_path = np.roll(close_path, 1)
            open_path[0] = start_price

            # Calculate the "Body" of the candle
            max_oc = np.maximum(open_path, close_path)
            min_oc = np.minimum(open_path, close_path)

            # Generate Wicks (Intraday Excursions)
            # We use the volatility of the SPECIFIC STATE for each day
            # If state is 'Panic', wicks will be huge. If 'Calm', wicks will be small.
            active_sigmas = np.array([sigmas[s] for s in synth_states])

            # We model the "Upper Wick" and "Lower Wick" as absolute normal distributions
            # The wick size is proportional to the daily volatility * Price
            # We use abs() because wicks only extend outwards
            upper_wicks = (
                # OLD (GAUSSIAN)
                # np.abs(
                #     np.random.normal(0, active_sigmas * wick_multiplier, size=length)
                # )
                np.abs(np.random.standard_t(df=self.tail_heaviness, size=length))
                * active_sigmas
                * wick_multiplier
                * close_path
            )
            lower_wicks = (
                # OLD (GAUSSIAN)
                # np.abs(
                #     np.random.normal(0, active_sigmas * wick_multiplier, size=length)
                # )
                np.abs(np.random.standard_t(df=self.tail_heaviness, size=length))
                * active_sigmas
                * wick_multiplier
                * close_path
            )

            high_path = max_oc + upper_wicks
            low_path = min_oc - lower_wicks

            return pd.DataFrame(
                {
                    "Open": open_path,
                    "High": high_path,
                    "Low": low_path,
                    "Close": close_path,
                },
                index=dates,
            )

    def _plot_spaghetti(self, real_curve, synth_curves):
        plt.figure(figsize=(12, 6))
        for c in synth_curves:
            plt.plot(range(len(c)), c.values, color="cyan", alpha=0.4)
        if real_curve is not None:
            plt.plot(
                range(len(real_curve)),
                real_curve.values,
                color="white",
                linewidth=2,
                label="Real History",
            )
        plt.title("Regime-Switching Monte Carlo (Regime switching generation)")
        plt.xlabel("Trading Days")
        plt.ylabel("Cumulative Log Return")
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

        # Regular walkforward test
        param_grid = [{"lookback": i} for i in range(200, 500, 20)]
        wf_evaluator = WalkForwardEvaluator(engine)
        wf_evaluator.evaluate(data, strat, param_grid)

        # Block bootstrapping walkforward test
        block_evaluator = BlockBoostrappingEvaluator(engine, wf_evaluator)
        block_evaluator.evaluate(data, strat, param_grid, n_runs=15, plots_block=False)

        # Regime switching evaluator
        regime_evaluator = RegimeSwitchingEvaluator(
            engine, wf_evaluator, realistic_ohlc=True, tail_heaviness=3
        )
        regime_evaluator.evaluate(data, strat, param_grid, n_runs=15)
