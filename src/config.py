from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kalshi
    kalshi_market_ticker: str = "FEDFUNDS"
    kalshi_poll_interval: int = 300
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"

    # Firecrawl
    firecrawl_api_key: str = ""
    firecrawl_search_queries: list[str] = Field(default_factory=lambda: [
        "federal reserve interest rate decision",
        "fed funds rate",
        "monetary policy",
        "FOMC",
    ])
    firecrawl_poll_interval: int = 1800
    firecrawl_max_articles: int = 10
    firecrawl_base_url: str = "https://api.firecrawl.dev/v1"

    # Jev / OpenRouter
    openrouter_api_key: str = ""
    jev_model: str = "typesafe/jev-1.13"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    vercel_ai_gateway_url: str = ""
    vercel_ai_gateway_token: str = ""
    jev_request_timeout: float = 30.0
    jev_max_retries: int = 3

    # Database
    database_path: Path = Path("data/kalshi_diffusion.db")

    # Diffusion model
    diffusion_window: int = 3600
    min_price_change: float = 0.01

    # Dashboard
    dashboard_output_dir: Path = Path("dashboard")
    dashboard_title: str = "Kalshi Information Diffusion Tracker"


settings = Settings()
