"""
Configuration management for the Stock Direction Predictor system.
Handles stock symbols, date ranges, model parameters, and system settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
import os
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data collection."""
    stock_symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMZN", "META"])
    start_date: str = "2020-01-01"
    end_date: str = "2025-09-30"
    data_source: str = "yahoo"
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    pattern_lengths: List[int] = field(default_factory=lambda: [3, 5, 7, 14])
    technical_indicators: List[str] = field(default_factory=lambda: [
        "RSI", "MACD", "EMA20", "EMA50", "EMA200", "ATR", "SMA"
    ])
    chart_patterns: List[str] = field(default_factory=lambda: [
        "golden_cross", "head_and_shoulder", "wedge"
    ])
    fibonacci_levels: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786])


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_types: List[str] = field(default_factory=lambda: ["xgboost", "random_forest", "svm", "neural_network"])
    train_test_split: float = 0.8
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_state: int = 42
    
    # XGBoost specific parameters
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })
    
    # Random Forest parameters
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    })
    
    # SVM parameters
    svm_params: Dict[str, Any] = field(default_factory=lambda: {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
    })
    
    # Neural Network parameters
    neural_network_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 1000
    })


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    position_size: float = 1.0  # Full position
    risk_free_rate: float = 0.02  # 2% annual


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    log_level: str = "INFO"
    log_file: str = "stock_predictor.log"
    model_save_path: str = "models"
    data_cache_path: str = "data_cache"
    results_path: str = "results"
    max_workers: int = 4
    memory_limit_gb: float = 8.0


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
    
    def load_config(self) -> Config:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self._config = self._dict_to_config(config_dict)
        else:
            self._config = Config()
            self.save_config()
        
        return self._config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        if self._config is None:
            self._config = Config()
        
        config_dict = self._config_to_dict(self._config)
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_config(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        if self._config is None:
            self._config = Config()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        config = Config()
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'features' in config_dict:
            config.features = FeatureConfig(**config_dict['features'])
        if 'models' in config_dict:
            config.models = ModelConfig(**config_dict['models'])
        if 'backtest' in config_dict:
            config.backtest = BacktestConfig(**config_dict['backtest'])
        if 'system' in config_dict:
            config.system = SystemConfig(**config_dict['system'])
        
        return config
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        def convert_value(value):
            """Convert values to YAML-serializable format."""
            if isinstance(value, tuple):
                return list(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            else:
                return value
        
        return {
            'data': {k: convert_value(v) for k, v in config.data.__dict__.items()},
            'features': {k: convert_value(v) for k, v in config.features.__dict__.items()},
            'models': {k: convert_value(v) for k, v in config.models.__dict__.items()},
            'backtest': {k: convert_value(v) for k, v in config.backtest.__dict__.items()},
            'system': {k: convert_value(v) for k, v in config.system.__dict__.items()}
        }


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration."""
    return config_manager.get_config()


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from specified path."""
    global config_manager
    config_manager = ConfigManager(config_path)
    return config_manager.load_config()