import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


def _message_signature(msg: dict[str, Any]) -> str:
    """Create a stable signature for a message based on role and content/tool_calls."""
    role = msg.get("role", "")
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    tool_call_id = msg.get("tool_call_id")
    payload = {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
    }
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()


def _messages_prefix_of(shorter: list[str], longer: list[str]) -> bool:
    """Check if `shorter` is a prefix of `longer` using message signatures."""
    if len(shorter) > len(longer):
        return False
    return all(s == l for s, l in zip(shorter, longer))


@dataclass
class SessionRecord:
    backend_session_id: str
    last_message_id: Any | None
    last_tools_signature: str | None
    last_message_count: int
    updated_at: float
    message_signatures: list[str] = field(default_factory=list)


class SessionManager:
    def __init__(
        self, create_backend_session: Callable[[], str], ttl_seconds: int = 7200
    ):
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

    def _signatures_for_messages(self, messages: list[dict[str, Any]]) -> list[str]:
        return [_message_signature(m) for m in messages]

    def _find_matching_session(self, messages: list[dict[str, Any]]) -> str | None:
        """Find the session whose message history is the longest prefix of `messages`."""
        sigs = self._signatures_for_messages(messages)
        best_match = None
        best_len = -1
        for sid, record in self._sessions.items():
            if _messages_prefix_of(record.message_signatures, sigs):
                if len(record.message_signatures) > best_len:
                    best_match = sid
                    best_len = len(record.message_signatures)
        return best_match

    def get_or_create(
        self,
        client_session_id: str | None,
        message_count: int,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str | None, Any | None, int]:
        with self._lock:
            self._cleanup_expired_locked()

            known = client_session_id and client_session_id in self._sessions

            if known:
                record = self._sessions[client_session_id]
                return (
                    client_session_id,
                    record.backend_session_id,
                    record.last_message_id,
                    record.last_message_count,
                )

            matched_id = None
            if messages:
                matched_id = self._find_matching_session(messages)

            if matched_id:
                record = self._sessions[matched_id]
                return (
                    matched_id,
                    record.backend_session_id,
                    record.last_message_id,
                    record.last_message_count,
                )

            backend_session_id = self._create_backend_session()
            return client_session_id, backend_session_id, None, 0

    def update(
        self,
        client_session_id: str,
        backend_session_id: str,
        last_message_id: Any | None,
        last_tools_signature: str | None = None,
        last_message_count: int | None = None,
        messages: list[dict[str, Any]] | None = None,
    ):
        with self._lock:
            previous = self._sessions.get(client_session_id)
            msg_sigs = (
                self._signatures_for_messages(messages)
                if messages
                else (previous.message_signatures if previous else [])
            )
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
                message_signatures=msg_sigs,
            )

    def new_client_session_id(self) -> str:
        return str(uuid.uuid4())

    def get_last_tools_signature(self, client_session_id: str | None) -> str | None:
        if not client_session_id:
            return None
        with self._lock:
            record = self._sessions.get(client_session_id)
            return record.last_tools_signature if record else None
