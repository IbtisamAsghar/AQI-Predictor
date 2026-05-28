import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App General configurations
    ENV: str = Field(default="development", description="Application runtime environment (development/production)")
    CITY: str = Field(default="Islamabad", description="Target city for weather and pollutant analysis")
    LATITUDE: float = Field(default=33.6844, description="Latitude coordinates of the target city")
    LONGITUDE: float = Field(default=73.0479, description="Longitude coordinates of the target city")
    
    # MongoDB Serverless Storage Settings
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017/aqi_db", 
        description="MongoDB connection string for Feature Store & Registry metadata"
    )
    
    # Hugging Face Hub Model Registry Settings
    HF_TOKEN: str = Field(default="", description="Hugging Face API write access token")
    HF_REPO_ID: str = Field(default="", description="Repository identifier on Hugging Face (e.g. username/repo)")
    
    # Live External APIs Tokens
    AQICN_TOKEN: str = Field(default="", description="Token for the AQICN World Air Quality index fallback API")
    
    # API Address Configuration (Dashboard serving)
    API_URL: str = Field(default="http://127.0.0.1:8000", description="Base URL of the FastAPI serving API")
    
    # Operational File Directories (Base relative calculations)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    MODEL_DIR: Path = BASE_DIR / "models"
    
    # Pydantic 2.x Settings Config Schema Loading
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    def create_directories(self) -> None:
        """Programmatically creates vital storage directories."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate settings singleton
settings = Settings()
settings.create_directories()

if __name__ == "__main__":
    # Self-test display
    print("Pearls AQI Settings successfully loaded:")
    print(f"  Environment:    {settings.ENV}")
    print(f"  Location Target: {settings.CITY} (Lat: {settings.LATITUDE}, Lon: {settings.LONGITUDE})")
    print(f"  Base directory:  {settings.BASE_DIR}")
    print(f"  Log directory:   {settings.LOG_DIR}")
    print(f"  Model directory: {settings.MODEL_DIR}")
    print(f"  MongoDB Status:  {'Configured (Atlas)' if 'mongodb+srv' in settings.MONGODB_URI else 'Default Local Fallback'}")
    print(f"  Hugging Face:    {'Authenticated' if settings.HF_TOKEN else 'Unauthenticated (Weights Registry limited)'}")
