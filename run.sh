#!/bin/bash

RUN=1
while true; do
    COMMIT=$(git rev-parse --short=6 HEAD)
    LOGFILE="agent_logs/agent_${COMMIT}.log"

    echo "=== Run #${RUN} | commit ${COMMIT} | $(date '+%Y-%m-%d %H:%M:%S') ==="

    claude --dangerously-skip-permissions \
           -p "$(cat AGENT_PROMPT.md)" \
           --model claude-opus-4-6 &> "$LOGFILE"

    echo "=== Run #${RUN} finished | $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo ""
    RUN=$((RUN + 1))
done
