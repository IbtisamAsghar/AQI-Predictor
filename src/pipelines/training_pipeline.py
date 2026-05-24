import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.utils.logging import get_logger
from src.data.feature_store import get_features_in_range
from src.models.model import AQIPredictionModel

logger = get_logger("training_pipeline")

def run_training_pipeline() -> bool:
    """Orchestrates feature store queries, preprocessing fits, training, and multi-horizon validation audits."""
    logger.info("="*60)
    logger.info("       STARTING AQI PREDICTOR MODEL RETRAINING PIPELINE")
    logger.info("="*60)
    
    try:
        # 1. Fetch all features in feature store
        start_date = "2020-01-01"
        end_date = "2030-12-31"
        df = get_features_in_range(start_date, end_date)
        
        if df.empty:
            logger.error("No historical records found in Feature Store. Retraining aborted.")
            return False
            
        # 2. Guarantee chronological sorting
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # 3. Identify target and metadata columns
        target_cols = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
        meta_cols = ["timestamp", "location", "latitude", "longitude"]
        
        # Separate input feature columns
        X_df = df.drop(columns=target_cols + meta_cols)
        feature_names = list(X_df.columns)
        
        logger.info(f"Loaded {df.shape[0]} samples with {len(feature_names)} active input features.")
        
        # 4. Strict Temporal chronological split (80% train, 20% validation)
        split_idx = int(df.shape[0] * 0.8)
        
        X_train_raw = X_df.iloc[:split_idx].values
        X_val_raw = X_df.iloc[split_idx:].values
        
        y_train_dict = {col: df[col].iloc[:split_idx].values for col in target_cols}
        y_val_dict = {col: df[col].iloc[split_idx:].values for col in target_cols}
        
        logger.info(f"Chronological Temporal Split:")
        logger.info(f"  Training Split:   {X_train_raw.shape[0]} rows")
        logger.info(f"  Validation Split: {X_val_raw.shape[0]} rows")
        
        # 5. Fit Preprocessing Pipeline ONLY on training features
        logger.info("Fitting preprocessing pipeline (Imputer + Scaler)...")
        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        X_train_scaled = preprocessor.fit_transform(X_train_raw)
        X_val_scaled = preprocessor.transform(X_val_raw)
        
        # 6. Fit multi-horizon regressor models
        logger.info("Training multi-horizon prediction estimators...")
        model = AQIPredictionModel(n_estimators=50, max_depth=8)
        model.fit(X_train_scaled, y_train_dict)
        logger.info("Models successfully trained.")
        
        # 7. Validation Performance Auditing
        logger.info("Evaluating predictions on temporal validation split...")
        preds_dict = model.predict(X_val_scaled)
        
        print("\n" + "="*70)
        print("          AQI FORECASTING MODEL PERFORMANCE METRICS REPORT")
        print("="*70)
        
        metrics = []
        for target in target_cols:
            y_val = y_val_dict[target]
            pred_col = target.replace("target", "pred")
            y_pred = preds_dict[pred_col]
            
            # Align predictions by removing NaN targets at chronological boundaries
            mask = ~np.isnan(y_val)
            y_val_clean = y_val[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_val_clean) == 0:
                logger.warning(f"No valid validation samples found for {target}. Skipping evaluation.")
                continue
                
            mae = mean_absolute_error(y_val_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_val_clean, y_pred_clean))
            r2 = r2_score(y_val_clean, y_pred_clean)
            
            horizon_label = target.split("_")[-1].upper() # 24H, 48H, 72H
            print(f"  Horizon {horizon_label} (+{horizon_label.lower()}):")
            print(f"    Mean Absolute Error (MAE):   {mae:.2f} AQI")
            print(f"    Root Mean Squared Error (RMSE): {rmse:.2f} AQI")
            print(f"    R² Variance Explained Score:   {r2:.4f}")
            print("-" * 50)
            
            metrics.append({
                "horizon": horizon_label,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            })
            
        print("="*70 + "\n")
        
        # 8. Save artifacts to models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model wrapper
        model_path = models_dir / "model.pkl"
        model.save(str(model_path))
        logger.info(f"Saved multi-horizon model wrapper to {model_path}")
        
        # Save fitted preprocessor pipeline
        preprocessor_path = models_dir / "preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Saved fitted preprocessor pipeline to {preprocessor_path}")
        
        # Save active feature name lists for FastAPI reference checks
        features_path = models_dir / "features.pkl"
        with open(features_path, "wb") as f:
            pickle.dump(feature_names, f)
        logger.info(f"Saved feature list definitions to {features_path}")
        
        # 9. Serverless Model Registry Publishing (Hugging Face Hub & MongoDB Atlas metadata)
        logger.info("Publishing trained artifacts and metadata to Model Registry...")
        from src.models.registry import register_model
        from datetime import timezone
        
        metrics_dict = {}
        for item in metrics:
            h_key = f"horizon_{item['horizon'].lower()}" # horizon_24h, horizon_48h, horizon_72h
            metrics_dict[h_key] = {
                "mae": float(item["mae"]),
                "rmse": float(item["rmse"]),
                "r2": float(item["r2"])
            }
            
        parameters_dict = {
            "n_estimators": int(model.n_estimators),
            "max_depth": int(model.max_depth),
            "random_state": int(model.random_state)
        }
        
        version_str = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        register_model(
            version=version_str,
            metrics=metrics_dict,
            parameters=parameters_dict,
            features=feature_names
        )
        
        logger.info("="*60)
        logger.info("   >>> MODEL TRAINING PIPELINE SUCCESSFUL & DEPLOYABLE <<<")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.critical(f"Model retraining pipeline execution crashed: {e}")
        raise e

if __name__ == "__main__":
    run_training_pipeline()
