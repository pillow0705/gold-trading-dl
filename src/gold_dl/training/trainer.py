"""
Training pipeline for gold prediction models
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from loguru import logger
from typing import Dict, Any


class ModelTrainer:
    """Train and optimize deep learning models"""

    def __init__(self, model_class, data_splits: Dict, config: Dict[str, Any]):
        self.model_class = model_class
        self.data_splits = data_splits
        self.config = config
        self.best_model = None

    def create_dataloaders(self, batch_size: int = 32):
        """Create PyTorch DataLoaders"""
        train_dataset = TensorDataset(
            torch.FloatTensor(self.data_splits['X_train']),
            torch.FloatTensor(self.data_splits['y_reg_train'])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.data_splits['X_val']),
            torch.FloatTensor(self.data_splits['y_reg_val'])
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(self.data_splits['X_test']),
            torch.FloatTensor(self.data_splits['y_reg_test'])
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def train(self, hyperparams: Dict[str, Any], max_epochs: int = 100):
        """Train model with given hyperparameters"""
        logger.info(f"Training with params: {hyperparams}")

        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            batch_size=hyperparams.get('batch_size', 32)
        )

        # Initialize model
        input_size = self.data_splits['X_train'].shape[2]
        model = self.model_class(input_size=input_size, **hyperparams)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='gold-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            mode='min'
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )

        # Logger
        tb_logger = TensorBoardLogger('logs', name='gold_model')

        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=tb_logger,
            accelerator='auto',
            devices=1
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Test
        test_results = trainer.test(model, test_loader)

        self.best_model = model
        return model, test_results

    def optimize_hyperparameters(self, n_trials: int = 20):
        """Optimize hyperparameters using Optuna"""
        def objective(trial):
            hyperparams = {
                'hidden_size': trial.suggest_int('hidden_size', 64, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
            }

            model, results = self.train(hyperparams, max_epochs=50)
            return results[0]['test_loss']

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        logger.success(f"Best params: {study.best_params}")
        return study.best_params
