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
    Saves a batch of records using highly robust, idempotent bulk upserts.
    If a record for the same timestamp and location already exists, it is overwritten.
    
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
    
    logger.info(f"Preparing bulk upserts for {len(records)} records...")
    operations = []
    
    for r in records:
        # Convert timestamp to string if passed as datetime
        ts = r["timestamp"]
        if isinstance(ts, datetime):
            ts = ts.isoformat()
            r["timestamp"] = ts
            
        loc = r["location"]
        
        # Build bulk operation matching compound key
        operations.append(
            UpdateOne(
                {"timestamp": ts, "location": loc},
                {"$set": r},
                upsert=True
            )
        )
        
    try:
        result = collection.bulk_write(operations, ordered=False)
        upserted_count = result.upserted_count
        modified_count = result.modified_count
        matched_count = result.matched_count
        
        logger.info(
            f"Successfully completed bulk writes. "
            f"Inserted (New): {upserted_count}, Modified (Overwritten): {modified_count}, "
            f"Matched (Unchanged): {matched_count - modified_count}."
        )
        return upserted_count + modified_count
        
    except Exception as e:
        logger.critical(f"Bulk upserts failed: {e}")
        raise RuntimeError("Feature store write error.") from e

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
        cursor = collection.find(query={}, projection={"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
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
