# ---------------------------- IMPORTS ---------------------------- #


import pandas as pd
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


# ---------------------------- BASE STRATEGY CLASS ---------------------------- #


class Strategy(ABC):
    """
    Abstract Base Class for all strategies.
    Forces every strategy to have a 'generate_signal' method.
    """

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ---------------------------- STRATEGIES ---------------------------- #


class DonchianBreakout(Strategy):
    @property
    def name(self):
        return "Donchian Breakout"

    def generate_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        lookback = params.get("lookback", 20)
        close = data["Close"]
        upper = close.rolling(window=lookback).max().shift(1)
        lower = close.rolling(window=lookback).min().shift(1)

        signal = pd.Series(0, index=data.index)
        signal[close > upper] = 1
        signal[close < lower] = -1
        return signal.ffill().fillna(0)


class MeanReversionZScore(Strategy):
    @property
    def name(self):
        return "Mean Reversion Z-Score"

    def generate_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        lookback = params.get("lookback", 20)
        entry_z = params.get("entry_z", 2.0)

        close = data["Close"]
        ma = close.rolling(window=lookback).mean()
        std = close.rolling(window=lookback).std()

        # Avoid division by zero
        std = std.replace(0, np.nan)
        z_score = (close - ma) / std

        signal = pd.Series(0, index=data.index)
        signal[z_score > entry_z] = -1
        signal[z_score < -entry_z] = 1
        return signal
