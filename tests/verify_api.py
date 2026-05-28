import sys
import time
import subprocess
import httpx
from pathlib import Path

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.database import get_db
from src.utils.logging import get_logger
from src.config import settings

logger = get_logger("verify_api")

def run_serving_audit():
    logger.info("="*70)
    logger.info("          PEARLS AQI PREDICTOR - REST API SERVING INTEGRATION AUDIT")
    logger.info("="*70)
    
    # 1. Start uvicorn server in a background subprocess
    logger.info("  1. Launching FastAPI microservice in background worker...")
    
    # Run uvicorn on localhost:8000
    cmd = [
        sys.executable, "-m", "uvicorn", "src.api.main:app", 
        "--host", "127.0.0.1", "--port", "8000"
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 2. Poll the server until it finishes dynamic model weights initialization
    logger.info("  2. Polling REST service until lifespan context weights loading completes...")
    
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=15.0)
    
    server_ready = False
    max_retries = 30
    retry_delay = 2
    
    for attempt in range(1, max_retries + 1):
        # First check if the subprocess has crashed
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            logger.error(f"     [FAIL] FastAPI background server crashed with exit code: {proc.poll()}")
            logger.error(f"     Stdout: {stdout}")
            logger.error(f"     Stderr: {stderr}")
            sys.exit(1)
            
        try:
            res = client.get("/health")
            if res.status_code == 200 and res.json().get("status") == "green":
                server_ready = True
                logger.info(f"     [SUCCESS] Connection established! Server is operational (Attempt {attempt}).")
                break
        except (httpx.ConnectError, httpx.HTTPError):
            logger.info(f"     Server not listening yet (Attempt {attempt}/{max_retries}). Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    if not server_ready:
        logger.error("     [FAIL] Serving layer failed to initialize and open port 8000 within 60 seconds.")
        proc.terminate()
        sys.exit(1)
    
    try:
        # 2. Test GET /health
        logger.info("  2. Auditing GET /health endpoint...")
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
        health_data = res_health.json()
        assert health_data["status"] == "green", f"Uhealthy server status: {health_data}"
        assert health_data["database"] == "connected", "Database connection failed in api context!"
        assert health_data["model_weights"] == "loaded", "Model weights failed to load into memory!"
        logger.info(f"     [SUCCESS] Status: {health_data['status']}, DB: {health_data['database']}, Version: {health_data['champion_version']}")
        
        # 3. Test GET /metrics
        logger.info("  3. Auditing GET /metrics endpoint...")
        res_metrics = client.get("/metrics")
        assert res_metrics.status_code == 200
        metrics_data = res_metrics.json()
        active_features = metrics_data["active_features"]
        logger.info(f"     [SUCCESS] Model Registry Version: {metrics_data['version']}")
        logger.info(f"     [SUCCESS] Hyperparameters Logs:  {metrics_data['parameters']}")
        logger.info(f"     [SUCCESS] Loaded Input Features:  {len(active_features)} features active")
        
        # 4. Test GET /predict/live
        logger.info("  4. Auditing GET /predict/live endpoint...")
        res_predict = client.get("/predict/live")
        assert res_predict.status_code == 200, f"Live predict failed: {res_predict.text}"
        pred_data = res_predict.json()
        assert "predictions" in pred_data
        assert "plus_24h" in pred_data["predictions"]
        assert "plus_48h" in pred_data["predictions"]
        assert "plus_72h" in pred_data["predictions"]
        logger.info(f"     [SUCCESS] Uptime Forecast successfully computed for Islamabad.")
        logger.info(f"     [SUCCESS] Current AQI: {pred_data['current_aqi']} ({pred_data['current_health_status']})")
        logger.info(f"     [SUCCESS] Predictions tomorrow (+24h): {pred_data['predictions']['plus_24h']}")
        logger.info(f"     [SUCCESS] Predictions day 2 (+48h):    {pred_data['predictions']['plus_48h']}")
        logger.info(f"     [SUCCESS] Predictions day 3 (+72h):    {pred_data['predictions']['plus_72h']}")
        
        # 5. Test POST /predict/custom
        logger.info("  5. Auditing POST /predict/custom endpoint...")
        
        # Query MongoDB Atlas to fetch a valid feature Store record as input
        db = get_db()
        latest_feature_record = db["features_hourly"].find_one(
            {"location": settings.CITY},
            sort=[("timestamp", -1)]
        )
        
        if not latest_feature_record:
            logger.error("     [FAIL] Could not query Feature Store for manual payload construction.")
            sys.exit(1)
            
        custom_payload = {f: latest_feature_record[f] for f in active_features}
        
        res_custom = client.post("/predict/custom", json={"features": custom_payload})
        assert res_custom.status_code == 200, f"Custom predict failed: {res_custom.text}"
        custom_data = res_custom.json()
        assert "predictions" in custom_data
        logger.info("     [SUCCESS] POST /predict/custom verified with exact live feature payload.")
        logger.info(f"     [SUCCESS] Custom +24h Forecast: {custom_data['predictions']['plus_24h']}")
        
    except AssertionError as e:
        logger.error(f"     [FAIL] Assertion failed: {e}")
        # Terminate server and exit
        proc.terminate()
        sys.exit(1)
    except Exception as e:
        logger.error(f"     [FAIL] API integration audit encountered exception: {e}")
        proc.terminate()
        sys.exit(1)
        
    # 6. Tear down background worker
    logger.info("  6. Tearing down FastAPI background worker...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        
    logger.info("="*70)
    logger.info("     >>> REST API SERVING AUDIT SUCCESSFUL - MICROSERVICE DEPLOYABLE <<<")
    logger.info("="*70)

if __name__ == "__main__":
    run_serving_audit()
