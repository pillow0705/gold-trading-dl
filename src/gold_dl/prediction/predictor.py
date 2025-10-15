"""Prediction and trading logic"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from loguru import logger


class GoldPredictor:
    """Make predictions and generate trading signals"""

    def __init__(self, model, scaler, config):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def generate_signals(self, predictions: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Generate buy/sell signals"""
        signals = np.zeros_like(predictions)
        signals[predictions > threshold] = 1  # Buy
        signals[predictions < -threshold] = -1  # Sell
        return signals

    def backtest(self, X_test: np.ndarray, y_test: np.ndarray, initial_capital: float = 10000):
        """Simple backtest"""
        predictions = self.predict(X_test)
        signals = self.generate_signals(predictions)

        portfolio = initial_capital
        positions = []

        for i, (signal, actual_return) in enumerate(zip(signals, y_test)):
            if signal == 1:  # Buy
                portfolio *= (1 + actual_return)
            elif signal == -1:  # Sell (short)
                portfolio *= (1 - actual_return)

            positions.append(portfolio)

        total_return = (portfolio - initial_capital) / initial_capital
        sharpe = np.mean(np.diff(positions)) / (np.std(np.diff(positions)) + 1e-8) * np.sqrt(252)

        logger.info(f"Backtest: Return={total_return:.2%}, Sharpe={sharpe:.2f}")

        return {
            'total_return': total_return,
            'final_portfolio': portfolio,
            'sharpe_ratio': sharpe,
            'positions': positions
        }
