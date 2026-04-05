# OpenAI Compatibility Proxy (v2)

## What this adds

- `proxy_server_v2.py`: Flask OpenAI-compatible `/v1/chat/completions` proxy over `ron`.
- `flatten.py`: deterministic flattening of OpenAI messages into one text prompt.
- `tool_parser.py`: robust tool-call JSON extraction + text cleanup.
- `session_manager.py`: in-memory client session mapping with 2-hour TTL.
- `test_openai_compat.py` + `conftest.py`: progressive 8-phase test suite.

## Requirements

- Python deps in `requirements.txt`
- Valid DeepSeek token (`DEEPSEEK_TOKEN`) or `--api-key`
- Optional `.env` file in project root containing `DEEPSEEK_TOKEN=...`

## Run server

```bash
python proxy_server_v2.py --api-key "$DEEPSEEK_TOKEN" --port 5005
```

Optional flags:

```bash
python proxy_server_v2.py --host 0.0.0.0 --port 5005 --debug --slowdown 1.5
```

## API endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Custom request field:

- `session_id` (optional): client-managed session ID returned by this proxy.

Implementation note: once a backend session is established, the proxy forwards only the latest
incremental turn (typically the final `user` or `tool` message) to avoid duplicating already-stored
history in the backend session.
User-message blocks are rendered without a `[USER]` tag for cleaner backend-visible conversation text.
If `tools` are unchanged from the previous turn in the same proxy session, tool definitions are not
re-sent to the backend prompt (they are sent again only when changed).

## Run tests

Run all:

```bash
pytest test_openai_compat.py -v --tb=short
```

Run by phase:

```bash
pytest test_openai_compat.py -m phase1 -v
pytest test_openai_compat.py -m phase3 -v
```

Stop after numbered test:

```bash
pytest test_openai_compat.py --stop-at 10 -v
```

Reuse previously passed tests (skip reruns), and persist pass state:

```bash
pytest test_openai_compat.py -v --reuse-passed
pytest test_openai_compat.py -v --reuse-passed --pass-log .pytest_passed_tests.json
```

Note: pass-state updates stop after the first failure in a run, so tests executed after a failing test are **not** marked as passed.
If you change test logic and want a clean rerun, delete the pass log file first (e.g. `rm -f .pytest_passed_tests.json`).

Name filter:

```bash
pytest test_openai_compat.py -k "test_1" -v
```
