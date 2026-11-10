import logging

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)

from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url=settings.lm_studio_base_url,
    api_key=settings.lm_studio_api_key,
    timeout=60.0,
    max_retries=0,
)


def generate_response(message: str) -> str:
    try:
        response = client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are LocalMind, a helpful local AI assistant. "
                        "Be accurate, clear and concise."
                    ),
                },
                {"role": "user", "content": message},
            ],
            temperature=0.7,
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError("The AI model returned an empty response.")

        return answer.strip()

    except APITimeoutError as error:
        raise RuntimeError("The AI model request timed out.") from error

    except APIConnectionError as error:
        raise RuntimeError(
            "Cannot connect to LM Studio. Check whether its server is running."
        ) from error

    except APIStatusError as error:
        logger.exception("LM Studio returned an API error.")
        raise RuntimeError(
            f"LM Studio returned an error (HTTP {error.status_code})."
        ) from error


def list_models() -> list[str]:
    models = client.models.list()
    return [model.id for model in models.data]