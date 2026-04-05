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

## Run server

```bash
python proxy_server_v2.py --api-key "$DEEPSEEK_TOKEN" --port 5005
```

Optional flags:

```bash
python proxy_server_v2.py --host 0.0.0.0 --port 5005 --debug
```

## API endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Custom request field:

- `session_id` (optional): client-managed session ID returned by this proxy.

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

Name filter:

```bash
pytest test_openai_compat.py -k "test_1" -v
```
