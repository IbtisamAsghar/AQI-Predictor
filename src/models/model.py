import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor

class AQIPredictionModel:
    """Wrapper class managing three independent regressors for multi-horizon AQI forecasts (+24h, +48h, +72h)."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        # Initialize independent estimators for each horizon
        self.models = {
            "target_aqi_24h": RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "target_aqi_48h": RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "target_aqi_72h": RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        }

    def _align_target(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filters out training samples that have NaN targets due to chronological boundary shifts."""
        mask = ~np.isnan(y)
        return X[mask], y[mask]

    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """
        Trains each independent regressor on the input feature matrix.
        
        Args:
            X: Standardized input feature matrix of shape (n_samples, n_features).
            y_dict: A dictionary containing targets for each forecasting horizon:
                    {"target_aqi_24h": y_24, "target_aqi_48h": y_48, "target_aqi_72h": y_72}
        """
        for horizon, model in self.models.items():
            if horizon not in y_dict:
                raise ValueError(f"Target values for horizon '{horizon}' were not provided.")
                
            y = y_dict[horizon]
            X_clean, y_clean = self._align_target(X, y)
            
            if len(y_clean) == 0:
                raise ValueError(f"No valid targets found to train model for horizon '{horizon}'.")
                
            model.fit(X_clean, y_clean)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates predictions for all three future horizons.
        
        Args:
            X: Standardized input feature matrix of shape (n_samples, n_features).
            
        Returns:
            Dictionary containing predictions:
            {"pred_aqi_24h": np.ndarray, "pred_aqi_48h": np.ndarray, "pred_aqi_72h": np.ndarray}
        """
        predictions = {}
        for horizon, model in self.models.items():
            # Standard random forest handles predictions continuously
            pred_col = horizon.replace("target", "pred")
            predictions[pred_col] = model.predict(X)
        return predictions

    def save(self, filepath: str):
        """Pickles and saves the trained multi-horizon model to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "AQIPredictionModel":
        """Loads and deserializes a saved model instance from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
