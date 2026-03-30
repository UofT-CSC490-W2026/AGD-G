from __future__ import annotations

import os
import time


def configure_hf_hub_timeouts(
    *,
    etag_timeout: int = 60,
    download_timeout: int = 120,
) -> None:
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(etag_timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))


def resolve_hf_token() -> str | None:
    for key in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGING_FACE_TOKEN",
        "API_KEY",
    ):
        value = os.getenv(key)
        if value:
            return value
    return None


def retry_hf_load(loader, *args, attempts: int = 3, sleep_seconds: float = 2.0, **kwargs):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return loader(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(sleep_seconds * attempt)
    raise last_exc
