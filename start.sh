#!/bin/bash
set -e

export PORT=${PORT:-8080}
export APP_MODE=${APP_MODE:-streamlit}

echo "Starting AML Fraud Detection app..."
echo "APP_MODE=${APP_MODE}"
echo "PORT=${PORT}"

if [ "${APP_MODE}" = "flask" ]; then
  echo "Launching Flask server on 0.0.0.0:${PORT}"
  exec python app.py
else
  echo "Launching Streamlit server on 0.0.0.0:${PORT}"
  exec streamlit run app_streamlit.py --server.port="${PORT}" --server.address=0.0.0.0
fi
