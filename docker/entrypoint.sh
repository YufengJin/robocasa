#!/usr/bin/env bash
set -e

cd /workspace

if [ $# -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
