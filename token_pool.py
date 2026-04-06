import logging
import threading
from typing import Callable

from ron.api import DeepSeekAPI

logger = logging.getLogger("token_pool")


class TokenPool:
    """Manages a round-robin pool of DeepSeek API tokens.

    Each new session created via ``_create_with_next()`` is assigned the next
    token in rotation and recorded in an internal mapping so that subsequent
    ``chat_completion`` calls can look up the correct API instance.

    Tokens that fail with an authorization error are automatically skipped and
    tried again on the next rotation.
    """

    def __init__(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("TokenPool requires at least one token")
        self._apis: list[DeepSeekAPI] = [DeepSeekAPI(token) for token in tokens]
        self._lock = threading.Lock()
        self._index = 0
        # Map backend session ID -> API instance that owns it.
        self._session_apis: dict[str, DeepSeekAPI] = {}
        # Track tokens that are known invalid (by API instance index).
        self._invalid_indices: set[int] = set()

    @property
    def size(self) -> int:
        return len(self._apis)

    def get_api_for_session(self, session_id: str) -> DeepSeekAPI | None:
        """Return the API instance associated with an existing backend session."""
        with self._lock:
            return self._session_apis.get(session_id)

    def _is_auth_error(self, exc: Exception) -> bool:
        """Detect whether an exception is caused by an invalid/expired token."""
        msg = str(exc).lower()
        return (
            "authorization" in msg
            or "invalid token" in msg
            or "auth" in msg
            or "40003" in msg
            or "401" in msg
        )

    def create_session_fn(self) -> Callable[[], str]:
        """Return a ``create_backend_session`` callable for ``SessionManager``."""
        return self._create_with_next

    def _create_with_next(self) -> str:
        """Advance the round-robin index and create a session with that API.

        If the chosen token fails with an authorization error, it is marked
        invalid and the next token is tried automatically.
        """
        total = len(self._apis)
        attempts = 0

        while attempts < total:
            with self._lock:
                idx = self._index
                self._index = (self._index + 1) % total
            attempts += 1

            if idx in self._invalid_indices:
                logger.debug("Skipping previously invalid token at index %d", idx)
                continue

            api = self._apis[idx]
            try:
                session_id = api.create_chat_session()
            except Exception as exc:
                if self._is_auth_error(exc):
                    logger.warning(
                        "Token at index %d failed auth (%s); skipping", idx, exc
                    )
                    with self._lock:
                        self._invalid_indices.add(idx)
                    continue
                raise  # non-auth error propagates

            with self._lock:
                self._session_apis[session_id] = api

            # Masked token prefix for identification (e.g. "sk-abc***" or "eyJhb***")
            raw_token = api.auth_token
            prefix = raw_token[:5] if raw_token else "?"
            logger.info(
                "[token_pool] session=%s token_index=%d token=%s***",
                session_id,
                idx,
                prefix,
            )
            return session_id

        raise RuntimeError(
            f"All {total} token(s) in the pool failed authorization. "
            "Check your DEEPSEEK_TOKEN* values in .env."
        )
