import sys
import time
from pathlib import Path
from threading import Lock
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("database_pool")

class MongoDBConnection:
    """Thread-safe Singleton class to manage MongoDB Atlas connection pools."""
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MongoDBConnection, cls).__new__(cls)
                cls._instance._client = None
            return cls._instance

    def connect(self) -> MongoClient:
        """Establishes and returns a connection client with automated retry logic."""
        if self._client is not None:
            return self._client

        retries = 3
        delay = 2
        
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"Connecting to MongoDB Atlas (Attempt {attempt}/{retries})...")
                # Instantiate MongoClient with robust pooling and timeout configurations
                import certifi
                client = MongoClient(
                    settings.MONGODB_URI,
                    tlsCAFile=certifi.where(),
                    serverSelectionTimeoutMS=5000,
                    maxPoolSize=50,
                    minPoolSize=5,
                    retryWrites=True
                )
                # Force connection check
                client.admin.command('ping')
                
                self._client = client
                logger.info("Successfully connected to MongoDB Atlas.")
                return self._client
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"Failed to connect to MongoDB on attempt {attempt}: {e}")
                if attempt == retries:
                    logger.critical("Failed to establish database connection after max retry limits.")
                    raise RuntimeError("Unable to connect to MongoDB Atlas feature store.") from e
                time.sleep(delay)
                delay *= 2

    def get_database(self):
        """Returns the target database reference."""
        client = self.connect()
        # Retrieve db name from connection string or fallback to default
        try:
            db_name = client.get_default_database().name
        except Exception:
            db_name = "aqi_db"
        return client[db_name]

    def close(self):
        """Safely closes the connection client."""
        if self._client:
            logger.info("Closing MongoDB connection pool.")
            self._client.close()
            self._client = None

# Export simple hook
db_client = MongoDBConnection()

def get_db():
    """Simple database retriever helper."""
    return db_client.get_database()
