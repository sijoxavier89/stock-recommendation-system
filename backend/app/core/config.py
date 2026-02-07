from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Stock Recommendation API"
    host: str = "127.0.0.1"
    port: int = 8000


settings = Settings()
