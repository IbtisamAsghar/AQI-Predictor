import sys
from pathlib import Path

# Add root folder to search path so tests can run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("sanity_check")

def run_verification() -> bool:
    """Executes basic assertions to ensure Phase 1 requirements are met."""
    logger.info("Starting Phase 1 Verification Checks...")
    
    # 1. Assert required operational directories were created
    logger.info("Checking operational directory creations...")
    assert settings.LOG_DIR.exists(), f"Log directory {settings.LOG_DIR} was not created!"
    assert settings.MODEL_DIR.exists(), f"Model directory {settings.MODEL_DIR} was not created!"
    logger.info("  [SUCCESS] Critical folders exist.")
    
    # 2. Check if the log file was successfully written to
    log_file_path = settings.LOG_DIR / "pipeline.log"
    assert log_file_path.exists(), "Log file 'pipeline.log' was not created!"
    logger.info("  [SUCCESS] Rotating log file successfully established.")
    
    # 3. Check loaded configuration schemas
    logger.info("Asserting environment configuration parameters...")
    assert settings.ENV in ["development", "production"], f"Invalid environment mode: {settings.ENV}"
    assert settings.CITY == "Islamabad", f"Expected target city Islamabad, got {settings.CITY}"
    assert isinstance(settings.LATITUDE, float) and isinstance(settings.LONGITUDE, float), "Coordinates must be floats!"
    logger.info("  [SUCCESS] Meteorological coordinates and variables verified.")
    
    # 4. Display loaded variables securely (mask tokens)
    masked_mongo = (
        "mongodb+srv://***:***@***" if "mongodb+srv" in settings.MONGODB_URI 
        else settings.MONGODB_URI
    )
    masked_hf_token = (
        f"hf_***{settings.HF_TOKEN[-4:]}" if len(settings.HF_TOKEN) > 6 
        else "NOT CONFIGRURED"
    )
    
    print("\n" + "="*70)
    print("           PEARLS AQI PREDICTOR - CONFIGURATION AUDIT REPORT")
    print("="*70)
    print(f"  Operational Environment: {settings.ENV.upper()}")
    print(f"  Target Location:         {settings.CITY} (Lat: {settings.LATITUDE}, Lon: {settings.LONGITUDE})")
    print(f"  Serverless Feature Store: {masked_mongo}")
    print(f"  Hugging Face Repository:  {settings.HF_REPO_ID or 'NOT CONFIGURED'}")
    print(f"  Hugging Face Write Token: {masked_hf_token}")
    print(f"  AQICN API Fallback:      {'Configured' if settings.AQICN_TOKEN else 'Not Configured'}")
    print(f"  Persistent Log File:     {log_file_path.resolve()}")
    print(f"  Model Storage Cache:     {settings.MODEL_DIR.resolve()}")
    print("="*70)
    print("  >>> PHASE 1 STATUS: SECURE AND FULLY OPERATIONAL <<<\n")
    
    return True

if __name__ == "__main__":
    try:
        success = run_verification()
        sys.exit(0 if success else 1)
    except AssertionError as ae:
        print(f"\n[CRITICAL ERROR] Phase 1 Verification Failed: {ae}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] Phase 1 crashed during check: {e}", file=sys.stderr)
        sys.exit(1)
