# autoppia_harvester

Minimal IWA-compatible harvester service.

It exposes:

- `GET /health`
- `POST /harvest`

`/harvest` accepts the task payload sent by `autoppia_iwa.src.web_agents.apified_harvester.ApifiedHarvester` and returns a harvested trajectory:

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

Run locally:

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 5060
```

Useful env vars:

- `AUTOPPIA_HARVESTER_CLAUDE_BIN` default `claude`
- `AUTOPPIA_HARVESTER_CLAUDE_MODEL` default `sonnet`
- `AUTOPPIA_HARVESTER_TIMEOUT_SECONDS` default `900`
- `AUTOPPIA_HARVESTER_WORKDIR` default `/tmp/autoppia_harvester`
- `AUTOPPIA_IWA_ROOT` default sibling `../autoppia_iwa`
