#!/bin/bash
# Overnight loop: check for new rounds every 15 minutes.
# Usage: nohup bash overnight.sh &
# Logs to: /home/penguin/astar/overnight.log

export PATH="$HOME/.local/bin:$PATH"
cd /home/penguin/astar

LOG="overnight.log"
echo "=== Overnight loop started at $(date -u) ===" >> "$LOG"

while true; do
    echo "" >> "$LOG"
    echo "--- Check at $(date -u) ---" >> "$LOG"
    uv run python auto_solve.py >> "$LOG" 2>&1
    echo "--- Sleeping 15 min ---" >> "$LOG"
    sleep 900
done
