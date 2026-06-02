# autoppia_harvester

Concrete IWA-compatible harvester service powered by Claude Code.

It exposes:

- `GET /health`
- `POST /find_trayectory`

`/find_trayectory` accepts the task payload sent by `autoppia_iwa.src.web_agents.apified_harvester.ApifiedHarvester` and returns a trajectory:

```json
{
  "web_agent_id": "autoppia-harvester",
  "trajectory": [
    {"name": "navigate", "arguments": {"url": "http://84.247.180.192:8000/"}}
  ],
  "actions": [
    {"name": "navigate", "arguments": {"url": "http://84.247.180.192:8000/"}}
  ],
  "model_used": "claude-code"
}
```

`POST /harvest` exists only as a deprecated alias while old callers migrate.

Run locally:

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 5060
```

Check subnet readiness:

```bash
python check.py
pytest
```

Useful env vars:

- `AUTOPPIA_HARVESTER_CLAUDE_BIN` default `claude`
- `AUTOPPIA_HARVESTER_CLAUDE_MODEL` default `sonnet`
- `AUTOPPIA_HARVESTER_TIMEOUT_SECONDS` default `900`
- `AUTOPPIA_HARVESTER_WORKDIR` default `/tmp/autoppia_harvester`
- `AUTOPPIA_IWA_ROOT` default sibling `../autoppia_iwa`
- `ANTHROPIC_API_KEY` or existing Claude Code auth
