module.exports = {
  apps: [
    {
      name: 'autoppia-operator-mcp',
      cwd: '/home/usuario1/autoppia/operator/autoppia_operator',
      script: '/home/usuario1/miniconda3/bin/python',
      args: '-m mcp.server',
      interpreter: 'none',
      env: {
        PYTHONUNBUFFERED: '1',
      },
      env_file: '/home/usuario1/autoppia/operator/autoppia_operator/.env',
      autorestart: true,
      max_restarts: 20,
      restart_delay: 2000,
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      out_file: '/tmp/autoppia-operator-mcp.out.log',
      error_file: '/tmp/autoppia-operator-mcp.err.log',
    },
  ],
};
