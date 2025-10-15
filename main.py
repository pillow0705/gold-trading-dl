#!/usr/bin/env python3
"""
Gold Trading Deep Learning Framework
Complete pipeline: Data → Training → Prediction → Trading
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from gold_dl.data.pipeline import GoldDataPipeline, DataConfig
from gold_dl.models.lstm_model import GoldLSTM, TransformerModel
from gold_dl.training.trainer import ModelTrainer
from gold_dl.prediction.predictor import GoldPredictor
from loguru import logger


def main():
    """Complete workflow demonstration"""
    logger.info("=== Gold Trading DL Framework ===")

    # 1. Data Pipeline
    logger.info("Step 1: Data Pipeline")
    config = DataConfig(
        symbol="GC=F",
        start_date="2015-01-01",
        sequence_length=60,
        prediction_horizon=1
    )

    pipeline = GoldDataPipeline(config)
    data_splits = pipeline.run_pipeline()

    # 2. Model Training
    logger.info("Step 2: Model Training")
    trainer = ModelTrainer(
        model_class=GoldLSTM,
        data_splits=data_splits,
        config=config.__dict__
    )

    hyperparams = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32
    }

    model, results = trainer.train(hyperparams, max_epochs=50)
    logger.success(f"Training completed: {results}")

    # 3. Prediction & Trading
    logger.info("Step 3: Prediction & Backtest")
    predictor = GoldPredictor(model, pipeline.scaler, config)

    backtest_results = predictor.backtest(
        data_splits['X_test'],
        data_splits['y_reg_test'],
        initial_capital=10000
    )

    logger.success(f"Backtest Return: {backtest_results['total_return']:.2%}")
    logger.success(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")

    print("\n=== Results ===")
    print(f"Final Portfolio: ${backtest_results['final_portfolio']:.2f}")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
