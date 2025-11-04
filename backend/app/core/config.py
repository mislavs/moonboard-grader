"""
Application configuration management using Pydantic Settings.

Loads configuration from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    app_name: str = Field(default="Moonboard Grade Predictor API", description="Application name")
    version: str = Field(default="1.0.0", description="API version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_allow_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed HTTP headers")
    
    # Model Configuration
    model_path: Path = Field(
        default=Path("models/model_for_inference.pth"),
        description="Path to the trained model file"
    )
    device: str = Field(default="cpu", description="Device for inference (cpu/cuda)")
    
    # Data Configuration
    problems_data_path: Path = Field(
        default=Path("../classifier/data/problems.json"),
        description="Path to the problems data JSON file"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Allow extra fields from environment
    )


# Global settings instance
settings = Settings()

