#!/usr/bin/env bash
# Builds the React dashboard and serves it on http://localhost:3001 (talks to the backend on :8000).
cd "$(dirname "$0")/../frontend"
npm run build >/dev/null 2>&1 || npm run build
exec npx vite preview --host 0.0.0.0 --port "${FRONTEND_PORT:-3001}" --strictPort
