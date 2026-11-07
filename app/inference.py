"""Small, reusable client for Ollama's local text-generation API."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """One generation attempt, including failures and available Ollama metrics."""

    model: str
    response: str | None
    wall_clock_seconds: float
    prompt_tokens: int | None
    generated_tokens: int | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_duration_ns: int | None
    eval_duration_ns: int | None
    status: str
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly mapping of this result."""
        return asdict(self)


class OllamaClient:
    """Send reproducible, non-streaming generation requests to Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def generate(self, model: str, prompt: str) -> InferenceResult:
        """Generate a response and return a structured result for any outcome."""
        if not model.strip():
            raise ValueError("model must not be empty")

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            return self._failure(model, started, "timeout", str(exc))
        except requests.ConnectionError as exc:
            return self._failure(model, started, "connection_error", str(exc))
        except requests.RequestException as exc:
            return self._failure(model, started, "request_error", str(exc))

        elapsed = time.perf_counter() - started
        try:
            payload = response.json()
        except (ValueError, requests.JSONDecodeError):
            payload = {}

        if response.status_code >= 400:
            detail = payload.get("error") if isinstance(payload, dict) else None
            message = str(detail or response.reason or "Ollama returned an HTTP error")
            error_type = "model_not_found" if response.status_code == 404 else "http_error"
            logger.warning("Ollama request failed (%s): %s", error_type, message)
            return InferenceResult(
                model=model,
                response=None,
                wall_clock_seconds=elapsed,
                prompt_tokens=None,
                generated_tokens=None,
                total_duration_ns=None,
                load_duration_ns=None,
                prompt_eval_duration_ns=None,
                eval_duration_ns=None,
                status="failed",
                error_type=error_type,
                error_message=message,
            )

        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            logger.warning("Ollama returned a malformed response for model %s", model)
            return InferenceResult(
                model=model,
                response=None,
                wall_clock_seconds=elapsed,
                prompt_tokens=None,
                generated_tokens=None,
                total_duration_ns=None,
                load_duration_ns=None,
                prompt_eval_duration_ns=None,
                eval_duration_ns=None,
                status="failed",
                error_type="invalid_response",
                error_message="Response JSON did not contain a string 'response' field",
            )

        return InferenceResult(
            model=model,
            response=payload["response"],
            wall_clock_seconds=elapsed,
            prompt_tokens=self._optional_int(payload.get("prompt_eval_count")),
            generated_tokens=self._optional_int(payload.get("eval_count")),
            total_duration_ns=self._optional_int(payload.get("total_duration")),
            load_duration_ns=self._optional_int(payload.get("load_duration")),
            prompt_eval_duration_ns=self._optional_int(payload.get("prompt_eval_duration")),
            eval_duration_ns=self._optional_int(payload.get("eval_duration")),
            status="success",
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _failure(
        model: str, started: float, error_type: str, error_message: str
    ) -> InferenceResult:
        logger.warning("Ollama request failed (%s): %s", error_type, error_message)
        return InferenceResult(
            model=model,
            response=None,
            wall_clock_seconds=time.perf_counter() - started,
            prompt_tokens=None,
            generated_tokens=None,
            total_duration_ns=None,
            load_duration_ns=None,
            prompt_eval_duration_ns=None,
            eval_duration_ns=None,
            status="failed",
            error_type=error_type,
            error_message=error_message,
        )