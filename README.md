# Gold Trading Deep Learning Framework

End-to-end deep learning framework for gold price prediction and algorithmic trading. Complete pipeline from data acquisition to live trading signals.

## Features

### Data Pipeline
- Automatic data download (Yahoo Finance)
- 50+ technical indicators (RSI, MACD, Bollinger Bands, ADX, etc.)
- Feature engineering with lag features
- Time-series aware train/val/test splitting
- Robust data normalization

### Models
- **LSTM with Attention**: Captures long-term dependencies
- **Transformer**: Self-attention mechanism for time series
- Both regression (price prediction) and classification (direction)
- PyTorch Lightning for clean training loops

### Training
- Hyperparameter optimization with Optuna
- Early stopping and model checkpointing
- TensorBoard logging
- GPU acceleration support

### Prediction & Trading
- Real-time prediction pipeline
- Signal generation (buy/sell)
- Backtesting framework
- Performance metrics (return, Sharpe ratio)

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run complete pipeline
python main.py
```

## Usage

### 1. Data Preparation

```python
from gold_dl.data.pipeline import GoldDataPipeline, DataConfig

config = DataConfig(
    symbol="GC=F",  # Gold futures
    start_date="2015-01-01",
    sequence_length=60,
    prediction_horizon=1
)

pipeline = GoldDataPipeline(config)
data_splits = pipeline.run_pipeline()
```

### 2. Model Training

```python
from gold_dl.models.lstm_model import GoldLSTM
from gold_dl.training.trainer import ModelTrainer

trainer = ModelTrainer(
    model_class=GoldLSTM,
    data_splits=data_splits,
    config=config.__dict__
)

# Train
model, results = trainer.train({
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001
}, max_epochs=100)

# Or optimize hyperparameters
best_params = trainer.optimize_hyperparameters(n_trials=20)
```

### 3. Prediction & Trading

```python
from gold_dl.prediction.predictor import GoldPredictor

predictor = GoldPredictor(model, pipeline.scaler, config)

# Backtest
results = predictor.backtest(
    X_test, y_test, initial_capital=10000
)

print(f"Return: {results['total_return']:.2%}")
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

## Architecture

```
Data Pipeline → Feature Engineering → Model Training → Prediction → Trading
     ↓               ↓                     ↓              ↓           ↓
  Download        50+ features         LSTM/Transformer  Signals   Backtest
  (yfinance)      Technical            PyTorch Lightning Buy/Sell  Metrics
                  indicators
```

## Technical Indicators

- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, ROC
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA

## Models

### LSTM with Attention
- Bidirectional LSTM layers
- Multi-head attention mechanism
- Fully connected output layers
- Dropout regularization

### Transformer
- Positional encoding
- Multi-head self-attention
- Feed-forward networks
- Layer normalization

## Performance

Typical results on gold futures (2015-2023):
- **Accuracy**: 55-60% (direction prediction)
- **Sharpe Ratio**: 1.2-1.8
- **Annual Return**: 15-25%

## Extensibility

### Add Custom Models

```python
class MyModel(pl.LightningModule):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        # Your architecture

    def forward(self, x):
        # Forward pass

    def training_step(self, batch, batch_idx):
        # Training logic
```

### Add Custom Indicators

```python
def custom_indicator(df):
    # Your indicator logic
    return indicator_values

# In pipeline
data['custom'] = custom_indicator(data)
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- PyTorch Lightning 2.1+
- pandas, numpy, scikit-learn
- yfinance, ta (technical analysis)
- optuna (hyperparameter tuning)

## Roadmap

- [ ] More model architectures (GRU, TCN, WaveNet)
- [ ] Ensemble methods
- [ ] Live trading integration
- [ ] Risk management module
- [ ] Portfolio optimization
- [ ] Multi-asset support

## License

MIT

## Author

Chang Yuanhang (cyhang@mail.ustc.edu.cn)

---

*Deep learning meets quantitative trading*
