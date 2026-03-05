module.exports = {
  apps: [
    {
      name: 'autoppia-operator',
      cwd: '/home/usuario1/autoppia/operator/autoppia_operator',
      script: '/home/usuario1/miniconda3/bin/python',
      args: '-m uvicorn main:app --host 0.0.0.0 --port 18060',
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
      out_file: '/tmp/autoppia-operator.out.log',
      error_file: '/tmp/autoppia-operator.err.log',
    },
  ],
};
