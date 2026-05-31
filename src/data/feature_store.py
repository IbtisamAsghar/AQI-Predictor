import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union
from pymongo import UpdateOne, ASCENDING, DESCENDING

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.database import get_db
from src.utils.logging import get_logger

logger = get_logger("feature_store")

# Define target collection
COLLECTION_NAME = "features_hourly"

def setup_indices():
    """Programmatically configures unique compound and fast retrieval indices."""
    db = get_db()
    collection = db[COLLECTION_NAME]
    
    logger.info("Initializing Feature Store database indices...")
    
    # 1. Compound unique index: {timestamp: 1, location: 1} to guarantee idempotence
    try:
        index_name = collection.create_index(
            [("timestamp", ASCENDING), ("location", ASCENDING)],
            unique=True,
            name="unique_timestamp_location"
        )
        logger.info(f"  [SUCCESS] Compound unique index created: {index_name}")
    except Exception as e:
        logger.error(f"Failed to create compound index: {e}")
        
    # 2. Descending timestamp index: {timestamp: -1} for high-performance serving reads
    try:
        index_name = collection.create_index(
            [("timestamp", DESCENDING)],
            name="serving_descending_timestamp"
        )
        logger.info(f"  [SUCCESS] Serving retrieval index created: {index_name}")
    except Exception as e:
        logger.error(f"Failed to create serving index: {e}")

def insert_records(records: List[Dict[str, Any]]) -> int:
    """
    Saves a batch of records using highly robust, idempotent, and chunked bulk upserts.
    If a record for the same timestamp and location already exists, it is overwritten.
    Large payloads are automatically split into smaller batches with retry backoffs.
    
    Args:
        records: A list of feature dictionaries.
        
    Returns:
        The count of successfully modified or inserted records.
    """
    if not records:
        logger.warning("Empty records list passed to insert_records. Skipping write.")
        return 0
        
    db = get_db()
    collection = db[COLLECTION_NAME]
    
    # Ensure indices are configured before write
    setup_indices()
    
    batch_size = 1000
    total_written = 0
    total_batches = (len(records) - 1) // batch_size + 1
    
    logger.info(f"Preparing chunked bulk upserts for {len(records)} records in {total_batches} batches...")
    
    import time
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        operations = []
        
        for r in batch:
            ts = r["timestamp"]
            if isinstance(ts, datetime):
                ts = ts.isoformat()
                r["timestamp"] = ts
                
            loc = r["location"]
            
            operations.append(
                UpdateOne(
                    {"timestamp": ts, "location": loc},
                    {"$set": r},
                    upsert=True
                )
            )
            
        retries = 3
        delay = 2
        batch_idx = i // batch_size + 1
        
        for attempt in range(1, retries + 1):
            try:
                result = collection.bulk_write(operations, ordered=False)
                upserted_count = result.upserted_count
                modified_count = result.modified_count
                total_written += (upserted_count + modified_count)
                
                logger.info(
                    f"  [Batch {batch_idx}/{total_batches}] Successful. "
                    f"Written: {upserted_count + modified_count} (New: {upserted_count}, Modified: {modified_count})."
                )
                break
            except Exception as e:
                logger.warning(f"  [Batch {batch_idx}/{total_batches}] Attempt {attempt} failed: {e}")
                if attempt == retries:
                    logger.critical(f"  [Batch {batch_idx}/{total_batches}] Failed permanently after {retries} attempts.")
                    raise RuntimeError("Feature store batch write error.") from e
                time.sleep(delay)
                delay *= 2
                
    logger.info(f"Successfully completed all bulk writes. Total upserted/modified: {total_written} records.")
    return total_written

def get_features_in_range(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieves chronological feature records inside a target window for model training.
    
    Args:
        start_date: ISO string or datetime starting boundary.
        end_date: ISO string or datetime ending boundary.
        
    Returns:
        Pandas DataFrame containing chronological records.
    """
    db = get_db()
    collection = db[COLLECTION_NAME]
    
    # Standardize boundaries to ISO string strings
    t_start = start_date.isoformat() if isinstance(start_date, datetime) else start_date
    t_end = end_date.isoformat() if isinstance(end_date, datetime) else end_date
    
    logger.info(f"Querying features in range: [{t_start}] -> [{t_end}]...")
    
    query = {
        "timestamp": {
            "$gte": t_start,
            "$lte": t_end
        }
    }
    
    try:
        # Fetch, sort ascending chronologically, and drop internal mongo _id
        cursor = collection.find(query, {"_id": 0}).sort("timestamp", ASCENDING)
        records = list(cursor)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            logger.info(f"Successfully retrieved {df.shape[0]} feature records.")
        else:
            logger.warning("Query returned zero feature store records.")
            
        return df
        
    except Exception as e:
        logger.error(f"Failed to query range: {e}")
        raise RuntimeError("Feature store range query error.") from e

def get_latest_features(limit: int = 49) -> pd.DataFrame:
    """
    Retrieves the newest N hours of features to compute rolling and lag transformations.
    
    Args:
        limit: The count of hours required (default 49 for rolling 24h & lag 48h checks).
        
    Returns:
        Pandas DataFrame sorted chronologically (oldest to newest).
    """
    db = get_db()
    collection = db[COLLECTION_NAME]
    
    logger.info(f"Querying the latest {limit} features for incremental engineering...")
    
    try:
        # Fetch using our high-performance descending index
        cursor = collection.find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
        records = list(cursor)
        
        df = pd.DataFrame(records)
        if not df.empty:
            # Reverse order so it flows chronologically (oldest to newest)
            df = df.iloc[::-1].reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            logger.info(f"Successfully fetched {df.shape[0]} newest feature records.")
        else:
            logger.warning("Latest features query returned zero records.")
            
        return df
        
    except Exception as e:
        logger.error(f"Failed to query newest records: {e}")
        raise RuntimeError("Feature store latest query error.") from e

if __name__ == "__main__":
    # Index initialization run
    setup_indices()
