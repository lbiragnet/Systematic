import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your existing data loader
from portfolio_engine import PortfolioDataLoader


class AlphaMiner:
    """
    An automated pipeline to generate mathematical features and evaluate
    their predictive power using the Information Coefficient (IC).
    """

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.returns = prices.pct_change()

    def _generate_features(self) -> dict:
        """
        Each feature must return a DataFrame of the same shape as self.prices.
        """
        print("Generating alpha features...")
        features = {}

        # Trend/Momentum Factors
        features["mom_1m"] = self.prices.pct_change(21)
        features["mom_3m"] = self.prices.pct_change(63)
        features["mom_6m"] = self.prices.pct_change(126)

        # Mean Reversion Factors
        # Formula: -(Price / SMA - 1). The negative sign aligns it so a higher score means "BUY".
        features["reversion_5d"] = -(self.prices / self.prices.rolling(5).mean() - 1)
        features["reversion_20d"] = -(self.prices / self.prices.rolling(20).mean() - 1)

        # Volatility / Risk Factors
        features["vol_5d"] = -self.returns.rolling(20).std()
        features["vol_20d"] = -self.returns.rolling(20).std()

        # Acceleration (Is momentum speeding up or slowing down?)
        features["acceleration_1m_3m"] = features["mom_1m"] - features["mom_3m"]

        return features

    def evaluate_alphas(self, forward_horizon: int = 5, verbose=True):
        """
        Calculates the Information Coefficient (IC) for all generated features.
        forward_horizon=5 means we are trying to predict the 5-day forward return.
        """
        if verbose:
            print(
                f"Evaluating Alphas (Predicting {forward_horizon}-day forward returns)..."
            )

        features = self._generate_features()

        # Create the TARGET: Shift future returns backwards to today
        # Target_t = (Price_{t+h} / Price_t) - 1
        target_returns = self.prices.pct_change(forward_horizon).shift(-forward_horizon)

        results = []

        for name, factor_df in features.items():
            # Calculate daily cross-sectional Spearman rank correlation
            # We compare the rank of the factor values to the rank of the future returns
            # Only calculate correlation on days where the factor has actual variance
            valid_rows = factor_df.std(axis=1) > 0
            clean_factor = factor_df[valid_rows]
            clean_targets = target_returns[valid_rows]
            daily_ic = clean_factor.corrwith(
                clean_targets, axis=1, method="spearman"
            ).dropna()

            # Calculate Metrics
            mean_ic = daily_ic.mean()
            ic_std = daily_ic.std()

            # Information Ratio (IR): Risk-adjusted IC. How consistent is the edge?
            ic_ir = mean_ic / ic_std if ic_std != 0 else 0

            # Win Rate: What % of days did this feature have a positive IC?
            win_rate = (daily_ic > 0).mean() * 100

            results.append(
                {
                    "Alpha Factor": name,
                    "Mean IC": mean_ic,
                    "IC IR": ic_ir,
                    "Win Rate (%)": win_rate,
                }
            )

        # Convert to DataFrame and sort by the absolute strength of the Mean IC
        results_df = pd.DataFrame(results)
        results_df["Abs IC"] = results_df["Mean IC"].abs()
        results_df = results_df.sort_values(by="Abs IC", ascending=False).drop(
            columns=["Abs IC"]
        )

        return results_df

    def combine_alphas(
        self, features_dict: dict, weights_and_directions: dict, verbose: bool = True
    ) -> pd.DataFrame:
        """
        Combines multiple alpha features into a single 'Super Alpha' using Z-scores.

        :param features_dict: A dictionary of {name: dataframe} (output of _generate_features)
        :param weights_and_directions: A dictionary defining the multiplier for each factor.
                                       e.g., {'mom_3m': 1.0, 'vol_20d': -1.0}
        """
        if verbose:
            print(f"Combining factors: {list(weights_and_directions.keys())}...")

        combined_score = pd.DataFrame(
            0.0, index=self.prices.index, columns=self.prices.columns
        )

        for name, weight in weights_and_directions.items():
            if name not in features_dict:
                print(f"⚠️ Warning: Factor '{name}' not found. Skipping.")
                continue

            factor_df = features_dict[name]

            # 1. Cross-Sectional Z-Score (Normalize each day across all assets)
            # Subtract the daily mean, divide by the daily standard deviation
            daily_mean = factor_df.mean(axis=1)
            daily_std = factor_df.std(axis=1)

            # Avoid division by zero on days where all assets have the exact same score
            daily_std = daily_std.replace(0, np.nan)

            z_scored_factor = factor_df.sub(daily_mean, axis=0).div(daily_std, axis=0)

            # 2. Apply direction and weight
            # If weight is -1.0, we invert the Z-score so higher remains "BUY"
            weighted_factor = z_scored_factor * weight

            # 3. Add to the master score
            combined_score = combined_score.add(weighted_factor, fill_value=0.0)

        return combined_score

    def optimize_combo_weights(
        self, features_dict: dict, factors_to_combine: list, forward_horizon: int = 5
    ):
        """
        Runs a Grid Search to find the optimal weight combination for a list of factors.
        """
        import itertools
        import warnings  # <-- NEW: Import warnings

        print(f"\nRunning Grid Search Optimization for: {factors_to_combine}")

        weight_grid = [-1.0, -0.5, 0.0, 0.5, 1.0]
        combinations = list(
            itertools.product(weight_grid, repeat=len(factors_to_combine))
        )

        target_returns = self.prices.pct_change(forward_horizon).shift(-forward_horizon)

        results = []

        # --- THE FIX: Identify valid rows for the TARGET once, outside the loop ---
        # We use > 1e-6 to avoid floating point math errors
        valid_targets = target_returns.std(axis=1) > 1e-6

        # We temporarily mute the Scipy warnings just for this calculation block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for combo in combinations:
                if all(w == 0 for w in combo):
                    continue

                weights_and_directions = dict(zip(factors_to_combine, combo))
                combined_score = self.combine_alphas(
                    features_dict, weights_and_directions, verbose=False
                )

                # --- THE FIX: Combine both valid masks ---
                valid_factors = combined_score.std(axis=1) > 1e-6
                valid_rows = valid_factors & valid_targets

                clean_score = combined_score[valid_rows]
                clean_targets = target_returns[valid_rows]

                # Evaluate it
                daily_ic = clean_score.corrwith(
                    clean_targets, axis=1, method="spearman"
                ).dropna()

                mean_ic = daily_ic.mean()
                ic_std = daily_ic.std()
                ir = mean_ic / ic_std if ic_std != 0 else 0

                weight_str = " | ".join(
                    [f"{f}: {w}" for f, w in weights_and_directions.items()]
                )

                results.append(
                    {
                        "Combination": weight_str,
                        "Mean IC": mean_ic,
                        "IC IR": ir,
                        "Win Rate (%)": (daily_ic > 0).mean() * 100,
                    }
                )

        results_df = pd.DataFrame(results).sort_values(by="IC IR", ascending=False)
        return results_df

    def neutralize_factor(
        self, factor_df: pd.DataFrame, sector_mapping: dict
    ) -> pd.DataFrame:
        """
        Applies Sector Neutralization.
        Transforms raw scores into Cross-Sectional Z-Scores strictly WITHIN each sector.
        """
        print("   -> Neutralizing factor against sector bias...")

        # 1. Melt the 2D matrix into a 1D list so we can group it easily
        stacked = factor_df.stack().reset_index()
        stacked.columns = ["Date", "Ticker", "Value"]

        # 2. Attach the sector labels
        stacked["Sector"] = stacked["Ticker"].map(sector_mapping)

        # 3. Define the internal Z-score math
        def intra_sector_zscore(x):
            if x.std() > 1e-6:
                return (x - x.mean()) / x.std()
            return x - x.mean()  # Return 0 if no variance

        # 4. Group by exactly Date AND Sector, then apply the math
        stacked["Neutral_Value"] = stacked.groupby(["Date", "Sector"])[
            "Value"
        ].transform(intra_sector_zscore)

        # 5. Pivot back into our standard 2D matrix format
        neutral_df = stacked.pivot(
            index="Date", columns="Ticker", values="Neutral_Value"
        )

        return neutral_df

    def combine_alphas_neutral(
        self, features_dict: dict, weights_and_directions: dict, verbose: bool = True
    ) -> pd.DataFrame:
        if verbose:
            print(
                f"Combining neutralized factors: {list(weights_and_directions.keys())}..."
            )

        combined_score = pd.DataFrame(
            0.0, index=self.prices.index, columns=self.prices.columns
        )

        for name, weight in weights_and_directions.items():
            if name not in features_dict:
                continue

            # The features_dict will now contain ALREADY neutralized Z-scores
            neutral_factor = features_dict[name]

            # Apply direction and weight
            weighted_factor = neutral_factor * weight

            # Add to the master score
            combined_score = combined_score.add(weighted_factor, fill_value=0.0)

        return combined_score


if __name__ == "__main__":
    plt.style.use("dark_background")

    sector_map = {
        # Technology & Communication
        "AAPL": "Tech",
        "MSFT": "Tech",
        "NVDA": "Tech",
        "GOOGL": "Tech",
        "META": "Tech",
        "AMZN": "Tech",
        "AVGO": "Tech",
        "CSCO": "Tech",
        "ORCL": "Tech",
        "CRM": "Tech",
        "AMD": "Tech",
        "INTC": "Tech",
        "TXN": "Tech",
        "QCOM": "Tech",
        "IBM": "Tech",
        # Financials
        "JPM": "Finance",
        "BAC": "Finance",
        "WFC": "Finance",
        "C": "Finance",
        "GS": "Finance",
        "MS": "Finance",
        "V": "Finance",
        "MA": "Finance",
        "AXP": "Finance",
        "SPGI": "Finance",
        # Healthcare
        "JNJ": "Health",
        "UNH": "Health",
        "LLY": "Health",
        "MRK": "Health",
        "ABBV": "Health",
        "PFE": "Health",
        "TMO": "Health",
        "DHR": "Health",
        "ABT": "Health",
        # Consumer
        "PG": "Consumer",
        "KO": "Consumer",
        "PEP": "Consumer",
        "WMT": "Consumer",
        "COST": "Consumer",
        "HD": "Consumer",
        "MCD": "Consumer",
        "NKE": "Consumer",
        # Industrials, Energy, Telecom
        "XOM": "Hard_Assets",
        "CVX": "Hard_Assets",
        "CAT": "Hard_Assets",
        "BA": "Hard_Assets",
        "UNP": "Hard_Assets",
        "T": "Hard_Assets",
        "VZ": "Hard_Assets",
        "DIS": "Hard_Assets",
    }

    # 50 Highly Liquid US Large-Cap Stocks
    universe = [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AMZN",
        "AVGO",
        "CSCO",
        "ORCL",
        "CRM",
        "AMD",
        "INTC",
        "TXN",
        "QCOM",
        "IBM",
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "V",
        "MA",
        "AXP",
        "SPGI",
        "JNJ",
        "UNH",
        "LLY",
        "MRK",
        "ABBV",
        "PFE",
        "TMO",
        "DHR",
        "ABT",
        "PG",
        "KO",
        "PEP",
        "WMT",
        "COST",
        "HD",
        "MCD",
        "NKE",
        "XOM",
        "CVX",
        "CAT",
        "BA",
        "UNP",
        "T",
        "VZ",
        "DIS",
    ]

    loader = PortfolioDataLoader()
    prices = loader.fetch_universe(universe, start_date="2016-01-01")

    if not prices.empty:
        miner = AlphaMiner(prices)
        features = miner._generate_features()

        # # Neutralise
        # neutral_features = {}
        # for factor_name, raw_df in features.items():
        #     print(f"Processing {factor_name}...")
        #     neutral_features[factor_name] = miner.neutralize_factor(raw_df, sector_map)

        # # 3. Run the Grid Search on the clean, neutralized data
        # factors_to_test = ["mom_6m", "vol_20d"]
        # optimization_results = miner.optimize_combo_weights(
        #     neutral_features, factors_to_test, forward_horizon=5
        # )

        # Run the Grid Search
        factors_to_test = ["mom_6m", "vol_20d"]
        optimization_results = miner.optimize_combo_weights(
            features, factors_to_test, forward_horizon=5
        )

        print("\n🏆 OPTIMIZATION LEADERBOARD (Sorted by IR) 🏆")
        print("=" * 80)
        print(
            optimization_results.head(10).to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )
        print("=" * 80)


# ==========================================
# MAIN EXECUTION
# ==========================================

# if __name__ == "__main__":
#     plt.style.use("dark_background")

#     # We use our standard Global Macro universe
#     universe = ["XLK", "XLE", "XLF", "TLT", "GLD", "BTC-USD"]

#     loader = PortfolioDataLoader()
#     prices = loader.fetch_universe(universe, start_date="2016-01-01")

#     if not prices.empty:
#         miner = AlphaMiner(prices)

#         # Test which factors predict the NEXT 5 DAYS of returns
#         leaderboard = miner.evaluate_alphas(forward_horizon=5)

#         print("\n🏆 ALPHA LEADERBOARD 🏆")
#         print("=" * 60)
#         # Format the DataFrame for clean printing
#         print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
#         print("=" * 60)

#         # Visualization
#         plt.figure(figsize=(10, 6))
#         # Sort values for the bar chart
#         chart_data = leaderboard.sort_values(by="Mean IC")
#         colors = ["#00ff00" if x > 0 else "#ff0000" for x in chart_data["Mean IC"]]

#         plt.barh(chart_data["Alpha Factor"], chart_data["Mean IC"], color=colors)
#         plt.axvline(0, color="white", linewidth=1)
#         plt.axvline(
#             0.02,
#             color="grey",
#             linestyle="--",
#             alpha=0.5,
#             label="Tradable Edge Threshold (+)",
#         )
#         plt.axvline(
#             -0.02,
#             color="grey",
#             linestyle="--",
#             alpha=0.5,
#             label="Tradable Edge Threshold (-)",
#         )

#         plt.title(
#             "Information Coefficient (IC) by Alpha Factor\n(Predicting 5-Day Forward Returns)"
#         )
#         plt.xlabel("Mean IC (Higher absolute value is better)")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


# if __name__ == "__main__":
#     plt.style.use("dark_background")

#     universe = [
#         # Technology & Communication
#         "AAPL",
#         "MSFT",
#         "NVDA",
#         "GOOGL",
#         "META",
#         "AMZN",
#         "AVGO",
#         "CSCO",
#         "ORCL",
#         "CRM",
#         "AMD",
#         "INTC",
#         "TXN",
#         "QCOM",
#         "IBM",
#         # Financials
#         "JPM",
#         "BAC",
#         "WFC",
#         "C",
#         "GS",
#         "MS",
#         "V",
#         "MA",
#         "AXP",
#         "SPGI",
#         # Healthcare
#         "JNJ",
#         "UNH",
#         "LLY",
#         "MRK",
#         "ABBV",
#         "PFE",
#         "TMO",
#         "DHR",
#         "ABT",
#         # Consumer Defensive & Cyclical
#         "PG",
#         "KO",
#         "PEP",
#         "WMT",
#         "COST",
#         "HD",
#         "MCD",
#         "NKE",
#         # Energy, Industrials & Telecom
#         "XOM",
#         "CVX",
#         "CAT",
#         "BA",
#         "UNP",
#         "T",
#         "VZ",
#         "DIS",
#     ]
#     loader = PortfolioDataLoader()
#     prices = loader.fetch_universe(universe, start_date="2016-01-01")

#     if not prices.empty:
#         miner = AlphaMiner(prices)

#         # 1. Generate all base features
#         features = miner._generate_features()

#         # 2. Combine our two best factors into a "Super Factor"
#         combo_settings = {"mom_3m": 0.5, "vol_20d": -1.5}

#         # 3. Create the new combined DataFrame and add it back to our features dictionary
#         features["SUPER_COMBO"] = miner.combine_alphas(features, combo_settings)

#         # 4. We need to slightly modify evaluate_alphas to accept our custom features dict
#         # (We will temporarily overwrite the miner's internal generation just for testing)
#         miner._generate_features = lambda: features

#         # 5. Evaluate everything together
#         leaderboard = miner.evaluate_alphas(forward_horizon=5)

#         print("\n🏆 ALPHA LEADERBOARD 🏆")
#         print("=" * 60)
#         print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
#         print("=" * 60)

#         # Visualization
#         plt.figure(figsize=(10, 6))
#         chart_data = leaderboard.sort_values(by="Mean IC")
#         colors = ["#00ff00" if x > 0 else "#ff0000" for x in chart_data["Mean IC"]]

#         # Highlight our combined factor in Cyan so it stands out
#         colors = [
#             "cyan" if name == "SUPER_COMBO" else c
#             for name, c in zip(chart_data["Alpha Factor"], colors)
#         ]

#         plt.barh(chart_data["Alpha Factor"], chart_data["Mean IC"], color=colors)
#         plt.axvline(0, color="white", linewidth=1)
#         plt.axvline(
#             0.02,
#             color="grey",
#             linestyle="--",
#             alpha=0.5,
#             label="Tradable Edge Threshold (+)",
#         )
#         plt.axvline(
#             -0.02,
#             color="grey",
#             linestyle="--",
#             alpha=0.5,
#             label="Tradable Edge Threshold (-)",
#         )

#         plt.title(
#             "Information Coefficient (IC) by Alpha Factor\n(Predicting 5-Day Forward Returns)"
#         )
#         plt.xlabel("Mean IC (Higher absolute value is better)")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
