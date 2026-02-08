#!/bin/bash
# XAUBot AI - Docker Logs Viewer

set -e
cd "$(dirname "$0")/../.."

SERVICE="$1"
LINES="${2:-100}"

if [ -z "$SERVICE" ]; then
    echo "📋 Viewing logs for all services..."
    echo "💡 Tip: Use './docker-logs.sh SERVICE [LINES]' to view specific service"
    echo "   Available: trading-api, dashboard, postgres, pgadmin"
    echo ""
    docker-compose logs -f --tail=$LINES
else
    echo "📋 Viewing logs for: $SERVICE (last $LINES lines)"
    echo ""
    docker-compose logs -f --tail=$LINES $SERVICE
fi
