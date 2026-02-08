#!/bin/bash
# XAUBot AI - Docker Start Script

set -e
cd "$(dirname "$0")/../.."

echo "🚀 Starting XAUBot AI Docker Services..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Creating .env from template..."
    cp docker/.env.docker.example .env
    echo "✅ .env created. Please edit it with your MT5 credentials:"
    echo "   - MT5_LOGIN"
    echo "   - MT5_PASSWORD"
    echo "   - MT5_SERVER"
    echo "   - MT5_PATH"
    echo ""
    read -p "Press Enter after editing .env to continue..."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Parse command line arguments
PROFILE_FLAG=""
if [ "$1" == "--admin" ] || [ "$1" == "-a" ]; then
    PROFILE_FLAG="--profile admin"
    echo "📊 Starting with pgAdmin (admin profile)..."
else
    echo "📊 Starting core services (postgres, api, dashboard)..."
    echo "💡 Use './docker-start.sh --admin' to include pgAdmin"
fi

echo ""

# Pull latest images
echo "📥 Pulling latest base images..."
docker-compose pull

echo ""
echo "🔨 Building services..."
docker-compose build

echo ""
echo "🎯 Starting services..."
docker-compose $PROFILE_FLAG up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check health
echo ""
echo "🏥 Checking service health..."
docker-compose ps

echo ""
echo "✅ Services started successfully!"
echo ""
echo "📍 Access Points:"
echo "   Dashboard:  http://localhost:3000"
echo "   API:        http://localhost:8000"
echo "   API Docs:   http://localhost:8000/docs"
echo "   Database:   localhost:5432"

if [ "$1" == "--admin" ] || [ "$1" == "-a" ]; then
    echo "   pgAdmin:    http://localhost:5050"
fi

echo ""
echo "📋 Useful Commands:"
echo "   View logs:       docker-compose logs -f"
echo "   View API logs:   docker-compose logs -f trading-api"
echo "   Stop services:   docker-compose down"
echo "   Restart:         docker-compose restart"
echo ""
echo "📖 Full documentation: docker/docs/DOCKER.md"
echo ""
