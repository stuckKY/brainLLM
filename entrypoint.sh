#!/bin/bash
set -e

echo "Starting brainLLM..."

# Start FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start Next.js frontend (HOSTNAME=0.0.0.0 ensures it binds to all interfaces,
# overriding Docker's default HOSTNAME which is the container ID)
cd /app/frontend && HOSTNAME=0.0.0.0 PORT=3000 node server.js &
FRONTEND_PID=$!

echo "Backend (PID $BACKEND_PID) on :8000"
echo "Frontend (PID $FRONTEND_PID) on :3000"

# If either process exits, shut down the other and exit
wait -n
EXIT_CODE=$?
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
exit $EXIT_CODE
