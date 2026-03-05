# Overview

The AlphaMiner is a (very basic) starting point for a statistical research engine. Its purpose is to discover, evaluate, and combine trading signals (which I will call features) before needing to conduct a full backtest. While a typical backtest tries to answer "Does this strategy make money in the past?", this engine is much simpler: "Does this specific formula have a statistically significant ability to rank future winners and losers?".

<br>

# Feature Engineering

A feature is a mathematical transformation of raw data (e.g. prices) designed to capture a specific market hypothesis. Some examples include:

1. Trend / Momentum
   - Hypothesis: assets that have gone up recently will continue to go up.
   - Expressed as the percentage return over the last $N$ days (for $N$-day momentum):
     $$ \text{Momentum}_N = \frac{P_t - P_{t-N}}{P_{t-N}} $$
   - Interpretation: a score of 0.05 means the asset grew 5% over $N$ days. 

2. Mean Reversion
   - Hypothesis: assets that have stretched too far from their average will snap back.
   - Expressed as the negative distance from the simple moving average (SMA). We multiply by $-1$ so that a drop in price creates a positive (BUY) signal:
     $$ \text{Mean-Reversion}_N = \frac{P_t}{SMA_N} - 1$$
   - Interpretation: a high score means the asset is heavily oversold relative to its recent average. 

3. Volatility
   - Hypothesis: the "low volatility anomaly" - boring, stable assets tend to outperform highly volatile, risky assets over time.
   - Expressed as the negative standard deviation of daily returns over $N$ days:
     $$ \text{Volatility}_N = -\sqrt{\frac{\sum_{i=1}^{N} (R_{t-i} - \bar{R})^2}{N-1}}$$
   - Interpretation: a high score (closer to zero, since it's negative) means the asset is very stable.

<br>

# Evaluation (Information Coefficient)

To gauge if a feature is effective, we calculate the Information Coefficient (IC). The IC measures the correlation between what our feature predicted and what actually happened. 

Crucially, we do not measure absolute returns; we measure rankings using the Spearman Rank Correlation. For every day, the engine performs the following steps: 

1. Rank the predictions ($R_x$): rank all stocks by their feature score today (e.g., AAPL is #1, MSFT is #50).
   
2. Rank the reality ($R_y$): look $H$ days into the future (forward horizon) and rank the stocks by their actual percentage return.
   
3. Calculate correlation ($\rho$): compare the two rankings using the Spearman formula, where $d_i$ is the difference in ranks for stock $i$, and $n$ is the number of stocks:
$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

The IC is computed each day:
$$IC_t = \rho(Rank(F_t), Rank(R_{t+h}))$$ 
We calculate its mean across the full dataset.

The Information Ratio (IR) refers to the mean IC divided by the standard deviation of the IC: 
$$IR = \bar{IC} / \sigma_{IC}$$
It is a risk-adjusted metric that decribes how consistent the signal is. A high average IC is not very useful if it is incredibly volatile. A high IR means the signal is smooth and reliable, not just driven by a few lucky days. This value is sometimes expressed on an annualised basis (e.g. multiplied by $\sqrt{252}$ for the number of trading days in a year).

<br>

# Combining Features (Z-Scoring)

If we want to test how a combination of features work together, we cannot simply add one to the other. For example, adding momentum (a percentage like 0.15) directly to volatility (a decimal like 0.02) makes no sense since they use completely different scales.

To combine features, we must standardize them into Z-Scores every day. For a specific day, we take a stock's raw feature score ($x$), subtract the daily average score of all stocks in our universe ($\mu$), and divide by the daily standard deviation of all stocks ($\sigma$).

$$Z = \frac{x - \mu}{\sigma}$$

A Z-score of +2.0 means "This stock's signal is 2 standard deviations stronger than the market average today." Because all features are now measured in the same unit (Standard Deviations), we can multiply them by a weight and add them together to create a combined feature.
$$Combined\_Score = (W_1 \times Z_1) + (W_2 \times Z_1)$$