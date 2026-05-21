import sys
from pathlib import Path

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.database import get_db
from src.data.feature_store import setup_indices, insert_records, get_features_in_range

def run_db_sanity_check() -> bool:
    """Performs deep verification of connection pool sanity, indexes, and unique constraints."""
    print("\n" + "="*70)
    print("          PEARLS AQI PREDICTOR - DATABASE INTEGRATION AUDIT")
    print("="*70)
    
    try:
        # 1. Retrieve connection reference and ping
        print("  1. Pinging MongoDB Atlas cluster... ", end="", flush=True)
        db = get_db()
        db.client.admin.command('ping')
        print("[SUCCESS]")
        
        # 2. Setup indexes
        print("  2. Ensuring collection indices are active... ", end="", flush=True)
        setup_indices()
        print("[SUCCESS]")
        
        # 3. Insert a temporary record
        print("  3. Testing database write permissions... ", end="", flush=True)
        test_timestamp = "2099-09-21T00:00:00Z"  # Synthetic timestamp for isolation
        test_record = {
            "timestamp": test_timestamp,
            "location": "TEST_ISOLATION_ZONE",
            "latitude": 33.0,
            "longitude": 73.0,
            "pm2_5": 10.0,
            "aqi": 42.0,
            "temperature": 25.0
        }
        
        written_count = insert_records([test_record])
        if written_count != 1:
            raise RuntimeError("Database reported write success but record was not registered.")
        print("[SUCCESS]")
        
        # 4. Read back the record
        print("  4. Testing chronological read queries... ", end="", flush=True)
        df = get_features_in_range(test_timestamp, test_timestamp)
        if df.empty:
            raise RuntimeError("Read query returned empty for the test isolation record.")
        
        # Assert values
        assert df.loc[0, "location"] == "TEST_ISOLATION_ZONE", "Location string mismatched on retrieval!"
        assert df.loc[0, "aqi"] == 42.0, "AQI float mismatched on retrieval!"
        print("[SUCCESS]")
        
        # 5. Assert Idempotency (Overwriting the exact same compound key)
        print("  5. Testing Unique Compound Index Idempotency... ", end="", flush=True)
        updated_record = test_record.copy()
        updated_record["aqi"] = 99.0  # Change AQI value to verify update
        
        insert_records([updated_record])
        df_updated = get_features_in_range(test_timestamp, test_timestamp)
        
        if df_updated.shape[0] != 1:
            raise AssertionError("Duplicate record detected! Compound key constraint failed!")
            
        assert df_updated.loc[0, "aqi"] == 99.0, "Upsert update failed to overwrite existing record!"
        print("[SUCCESS]")
        
        # 6. Safe database cleanup
        print("  6. Purging temporary test records... ", end="", flush=True)
        collection = db["features_hourly"]
        del_result = collection.delete_many({"location": "TEST_ISOLATION_ZONE"})
        print(f"[SUCCESS] (Deleted {del_result.deleted_count} items)")
        
        print("="*70)
        print("  >>> DATABASE CONNECTION & OPERATORS: 100% OPERATIONAL <<<")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"[CRITICAL FAILURE]\n\nDetails: {e}")
        print("="*70 + "\n")
        return False

if __name__ == "__main__":
    success = run_db_sanity_check()
    sys.exit(0 if success else 1)
