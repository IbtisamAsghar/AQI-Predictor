import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config import settings

def setup_logging() -> None:
    """Sets up the global application-level logging systems."""
    # Ensure standard logs directory exists
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = settings.LOG_DIR / "pipeline.log"
    
    # Retrieve the root logger reference
    root_logger = logging.getLogger()
    
    # Prevent duplicated handlers if initialized multiple times
    if root_logger.handlers:
        return
        
    # Set default minimum level to INFO
    root_logger.setLevel(logging.INFO)
    
    # Standardized MLOps structured format definition
    log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    
    # 1. Console Stream Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. Disk Rotating File Handler (Max 10MB per file, rotating 5 versions)
    try:
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10 Megabytes
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"CRITICAL WARNING: Unable to initialize file logging handler: {e}")

# Trigger standard logging setup upon module import
setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Helper to return a configured child logger instance."""
    return logging.getLogger(name)

if __name__ == "__main__":
    # Internal validation test run
    logger = get_logger("logging_system")
    logger.info("Pearls AQI Logger is active and configured.")
    logger.warning("Verify that log files write successfully under 'logs/pipeline.log'.")
