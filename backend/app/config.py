from pathlib import Path
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    lm_studio_base_url: str
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("lm_studio_base_url")
    @classmethod
    def validate_lm_studio_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parsed_url = urlparse(value)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError("LM Studio URL must use HTTP or HTTPS.")

        if not parsed_url.netloc:
            raise ValueError("LM Studio URL must include a valid host.")

        return value

    @field_validator("lm_studio_model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("LM Studio model name cannot be empty.")

        return value


settings = Settings()
