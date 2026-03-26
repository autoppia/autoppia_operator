# MCP Package

Primary MCP implementation now lives in `mcp/`.

## Run modes

- Direct:
  - `python -m mcp.server`
- List tools:
  - `python -m mcp.server --list-tools`
- Backward-compatible path (still works):
  - `python mcp/server.py`

## PM2

Use the canonical PM2 config:

```bash
pm2 start deploy/pm2/mcp.config.cjs
pm2 logs autoppia-operator-mcp
```

## MCP client config

Use module entrypoint or script path:

```json
{
  "mcpServers": {
    "miner_mcp": {
      "command": "python",
      "args": ["/home/usuario1/autoppia/operator/autoppia_operator/mcp/server.py"],
      "env": {
        "SN36_NETWORK": "finney",
        "SN36_NETUID": "36"
      }
    }
  }
}
```
