"""
Core interfaces for the Stock Direction Predictor system.
Defines abstract base classes for all major components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StockData:
    """Data model for stock information."""
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float


@dataclass
class TechnicalIndicators:
    """Data model for technical indicators."""
    rsi: float
    macd: float
    macd_signal: float
    ema_20: float
    ema_50: float
    ema_200: float
    atr: float
    sma: float


@dataclass
class CandlestickPattern:
    """Data model for candlestick patterns."""
    pattern_length: int
    signal: int  # -1, 0, 1
    confidence: float
    pattern_type: str


@dataclass
class ModelConfiguration:
    """Data model for model configuration."""
    model_type: str
    pattern_length: int
    hyperparameters: Dict[str, Any]
    feature_set: List[str]
    version: str


@dataclass
class BacktestResult:
    """Data model for backtesting results."""
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    trade_log: pd.DataFrame
    portfolio_values: pd.Series


class IDataCollectionService(ABC):
    """Interface for data collection services."""
    
    @abstractmethod
    def fetch_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch historical stock data for given symbols and date range."""
        pass
    
    @abstractmethod
    def validate_data_completeness(self, data: pd.DataFrame) -> bool:
        """Validate that the data contains all required fields and is complete."""
        pass
    
    @abstractmethod
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        pass


class IFeatureEngineeringModule(ABC):
    """Interface for feature engineering modules."""
    
    @abstractmethod
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given data."""
        pass
    
    @abstractmethod
    def generate_candlestick_signals(self, data: pd.DataFrame, pattern_length: int) -> pd.DataFrame:
        """Generate candlestick pattern signals for the given pattern length."""
        pass
    
    @abstractmethod
    def detect_chart_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect chart patterns in the data."""
        pass
    
    @abstractmethod
    def calculate_fibonacci_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels."""
        pass


class ICandlestickPatternGenerator(ABC):
    """Interface for candlestick pattern generators."""
    
    @abstractmethod
    def generate_n_day_signals(self, data: pd.DataFrame, n: int) -> pd.Series:
        """Generate N-day candlestick pattern signals."""
        pass
    
    @abstractmethod
    def validate_pattern_consistency(self, signals: pd.Series) -> bool:
        """Validate the consistency of generated patterns."""
        pass


class IMLModel(ABC):
    """Interface for machine learning models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        pass


class IModelTrainingPipeline(ABC):
    """Interface for model training pipelines."""
    
    @abstractmethod
    def prepare_training_data(self, features: pd.DataFrame, targets: pd.Series, pattern_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the given pattern length."""
        pass
    
    @abstractmethod
    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> IMLModel:
        """Train a model of the specified type."""
        pass
    
    @abstractmethod
    def validate_model(self, model: IMLModel, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Validate the model and return performance metrics."""
        pass
    
    @abstractmethod
    def save_model(self, model: IMLModel, model_id: str) -> str:
        """Save the model and return the saved model path."""
        pass


class IBacktestingEngine(ABC):
    """Interface for backtesting engines."""
    
    @abstractmethod
    def simulate_trading(self, signals: pd.Series, prices: pd.Series, initial_capital: float) -> BacktestResult:
        """Simulate trading based on signals and return results."""
        pass
    
    @abstractmethod
    def calculate_portfolio_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        pass
    
    @abstractmethod
    def generate_trade_log(self, signals: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """Generate detailed trade log."""
        pass


class IPerformanceEvaluator(ABC):
    """Interface for performance evaluation."""
    
    @abstractmethod
    def calculate_prediction_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction accuracy metrics (MSE, MAE, RMSE)."""
        pass
    
    @abstractmethod
    def calculate_financial_metrics(self, backtest_result: BacktestResult) -> Dict[str, float]:
        """Calculate financial performance metrics."""
        pass
    
    @abstractmethod
    def rank_model_combinations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank model-pattern combinations by performance."""
        pass