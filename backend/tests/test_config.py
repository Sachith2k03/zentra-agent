import pytest
from pydantic import ValidationError

from app.config import Settings


def test_valid_settings():
    settings = Settings(
        _env_file=None,
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_model="qwen3.5-4b",
    )

    assert settings.lm_studio_base_url == "http://localhost:1234/v1"
    assert settings.lm_studio_model == "qwen3.5-4b"


def test_base_url_removes_trailing_slash():
    settings = Settings(
        _env_file=None,
        lm_studio_base_url="http://localhost:1234/v1/",
        lm_studio_model="qwen3.5-4b",
    )

    assert settings.lm_studio_base_url == "http://localhost:1234/v1"


def test_invalid_base_url():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            lm_studio_base_url="ftp://localhost:1234/v1",
            lm_studio_model="qwen3.5-4b",
        )


def test_empty_model_name():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            lm_studio_base_url="http://localhost:1234/v1",
            lm_studio_model="   ",
        )


def test_missing_model_name():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            lm_studio_base_url="http://localhost:1234/v1",
        )
