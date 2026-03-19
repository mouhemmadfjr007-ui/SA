module.exports = {
  apps: [{
    name: 'sentiment-arabia',
    script: 'python3',
    args: '-m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1',
    cwd: '/home/user/webapp',
    env: {
      PYTHONPATH: '/home/user/webapp',
      JWT_SECRET: 'sentiment-arabia-2024'
    },
    watch: false,
    instances: 1,
    exec_mode: 'fork'
  }]
}
