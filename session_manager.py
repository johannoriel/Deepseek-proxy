import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SessionRecord:
    backend_session_id: str
    last_message_id: Any | None
    last_tools_signature: str | None
    last_message_count: int
    updated_at: float


class SessionManager:
    def __init__(self, create_backend_session: Callable[[], str], ttl_seconds: int = 7200):
        self._create_backend_session = create_backend_session
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionRecord] = {}

    def _is_expired(self, record: SessionRecord) -> bool:
        return (time.time() - record.updated_at) > self._ttl_seconds

    def _cleanup_expired_locked(self) -> None:
        expired = [sid for sid, rec in self._sessions.items() if self._is_expired(rec)]
        for sid in expired:
            del self._sessions[sid]

    def get_or_create(self, client_session_id: str | None, message_count: int) -> tuple[str, Any | None, int]:
        with self._lock:
            self._cleanup_expired_locked()

            known = client_session_id and client_session_id in self._sessions
            force_new = message_count == 1 and not known

            if not client_session_id or not known or force_new:
                backend_session_id = self._create_backend_session()
                return backend_session_id, None, 0

            record = self._sessions[client_session_id]
            return record.backend_session_id, record.last_message_id, record.last_message_count

    def update(
        self,
        client_session_id: str,
        backend_session_id: str,
        last_message_id: Any | None,
        last_tools_signature: str | None = None,
        last_message_count: int | None = None,
    ):
        with self._lock:
            previous = self._sessions.get(client_session_id)
            self._sessions[client_session_id] = SessionRecord(
                backend_session_id=backend_session_id,
                last_message_id=last_message_id,
                last_tools_signature=last_tools_signature
                if last_tools_signature is not None
                else (previous.last_tools_signature if previous else None),
                last_message_count=last_message_count
                if last_message_count is not None
                else (previous.last_message_count if previous else 0),
                updated_at=time.time(),
            )

    def new_client_session_id(self) -> str:
        return str(uuid.uuid4())

    def get_last_tools_signature(self, client_session_id: str | None) -> str | None:
        if not client_session_id:
            return None
        with self._lock:
            record = self._sessions.get(client_session_id)
            return record.last_tools_signature if record else None
