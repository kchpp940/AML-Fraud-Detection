#!/bin/bash
set -e

export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8080}
export APP_MODE=${APP_MODE:-streamlit}

echo "Starting AML Fraud Detection app..."
echo "APP_MODE=${APP_MODE}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"

if [ "${APP_MODE}" = "flask" ]; then
  echo "Launching Flask server on ${HOST}:${PORT}"
  exec python app.py
else
  echo "Launching Streamlit server on ${HOST}:${PORT}"
  exec streamlit run app_streamlit.py --server.address="${HOST}" --server.port="${PORT}"
fi
