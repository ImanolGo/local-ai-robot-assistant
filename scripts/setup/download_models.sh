#!/bin/bash
# Quick model download helper script
# Usage: ./download_models.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🤖 Local AI Robot Assistant - Model Download Helper"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

# Change to project directory
cd "$PROJECT_ROOT"

# Run the download script with all arguments passed through
python3 scripts/setup/download_models.py "$@"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model download completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Convert models to TensorRT: python tools/convert_yolo.py"
    echo "   2. Convert depth model: python tools/convert_depth.py"
    echo "   3. Set up LLM with NanoLLM"
    echo ""
    echo "💾 Models location: $PROJECT_ROOT/models/"
    echo "📖 Documentation: docs/model_credits.md"
else
    echo ""
    echo "❌ Model download failed. Check the output above for details."
    exit 1
fi
