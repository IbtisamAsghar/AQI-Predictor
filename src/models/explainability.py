import shap
import numpy as np
from typing import Dict, Any, List
from src.utils.logging import get_logger

logger = get_logger("explainability")

class AQIExplainer:
    """Manages SHAP TreeExplainer instances for the multi-horizon forecasting models."""
    
    def __init__(self, model_wrapper, preprocessor, feature_names: List[str]):
        """
        Initializes the explainer with pre-loaded models and scaler.
        
        Args:
            model_wrapper: Instance of AQIPredictionModel containing estimators.
            preprocessor: StandardScaler/MinMaxScaler fitted preprocessor.
            feature_names: List of strings mapping feature column names.
        """
        self.model_wrapper = model_wrapper
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
        # Initialize shap TreeExplainers for each Random Forest estimator
        self.explainers = {}
        for horizon, model in model_wrapper.models.items():
            try:
                # We use tree_path_dependent perturbation which is very fast and requires no background data
                self.explainers[horizon] = shap.TreeExplainer(
                    model,
                    feature_perturbation="tree_path_dependent"
                )
                logger.info(f"Initialized SHAP TreeExplainer for horizon: {horizon}")
            except Exception as e:
                logger.error(f"Failed to initialize SHAP TreeExplainer for {horizon}: {e}")

    def explain_instance(self, X_scaled: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Computes SHAP values for a single scaled input row across all horizons.
        
        Args:
            X_scaled: numpy array of shape (1, n_features)
            
        Returns:
            Dictionary containing explanation details for each horizon:
            {
               "plus_24h": {
                   "base_value": float,
                   "prediction": float,
                   "contributions": {feature_name: shap_value, ...}
               },
               ...
            }
        """
        explanations = {}
        
        for horizon, explainer in self.explainers.items():
            horizon_key = horizon.replace("target_aqi_", "plus_")
            try:
                if horizon not in self.explainers:
                    raise ValueError(f"TreeExplainer not initialized for {horizon}")
                    
                # Compute SHAP values for this instance
                shap_values = explainer.shap_values(X_scaled)
                
                # Check for list structures (sometimes returned in shap if dimensions vary)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # shap_values shape for regressor on 1 row is (1, n_features)
                # Flatten to 1D array of shape (n_features,)
                shap_flat = shap_values.flatten()
                
                # Expected value is a numpy array or single float depending on shap version
                base_value = explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0]
                base_value = float(base_value)
                
                # Map SHAP values to feature names
                contributions = {}
                for name, val in zip(self.feature_names, shap_flat):
                    contributions[name] = float(val)
                    
                # Compute prediction sum
                pred = base_value + np.sum(shap_flat)
                
                explanations[horizon_key] = {
                    "base_value": base_value,
                    "prediction": pred,
                    "contributions": contributions
                }
            except Exception as e:
                logger.error(f"Error computing SHAP values for {horizon}: {e}")
                # Fallback to scikit-learn feature importances signature or zeroed dicts
                explanations[horizon_key] = {
                    "base_value": 0.0,
                    "prediction": 0.0,
                    "contributions": {},
                    "error": str(e)
                }
                
        return explanations

    def get_global_importances(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves global feature importances based on Random Forest feature importances
        as a fallback/overall metric.
        
        Returns:
            {
               "plus_24h": [{"feature": name, "importance": score}, ...],
               ...
            }
        """
        global_importances = {}
        for horizon, model in self.model_wrapper.models.items():
            horizon_key = horizon.replace("target_aqi_", "plus_")
            try:
                importances = model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]
                
                feature_scores = [
                    {"feature": self.feature_names[idx], "importance": float(importances[idx])}
                    for idx in sorted_indices
                ]
                global_importances[horizon_key] = feature_scores
            except Exception as e:
                logger.error(f"Error getting feature importances for {horizon}: {e}")
                global_importances[horizon_key] = []
        return global_importances
