#!/bin/bash
# XAUBot AI - Docker Stop Script

set -e
cd "$(dirname "$0")/../.."

echo "🛑 Stopping XAUBot AI Docker Services..."
echo ""

# Parse arguments
if [ "$1" == "--remove" ] || [ "$1" == "-r" ]; then
    echo "⚠️  Stopping and removing containers..."
    docker-compose down
    echo "✅ Containers stopped and removed"
elif [ "$1" == "--clean" ] || [ "$1" == "-c" ]; then
    echo "⚠️  WARNING: This will remove all data including database!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" == "yes" ]; then
        docker-compose down -v
        echo "✅ Containers, networks, and volumes removed"
    else
        echo "❌ Cancelled"
    fi
else
    echo "🔄 Stopping containers (data will be preserved)..."
    docker-compose stop
    echo "✅ Containers stopped"
fi

echo ""
echo "📋 To restart: ./docker/scripts/docker-start.sh"
echo ""
