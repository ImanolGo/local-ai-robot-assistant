#!/bin/bash
# Test Reorganization Script
# Moves integration tests from package directories to integration_tests/

set -e

echo "🔄 Reorganizing test structure for better separation of concerns..."

# Create backup
echo "📦 Creating backup of current test structure..."
cp -r src/ src_backup_$(date +%Y%m%d_%H%M%S)

# Move integration tests from perception_nodes to integration_tests/
echo "📦→🔗 Moving perception_nodes integration tests..."
mv src/perception_nodes/test/test_camera_pipeline_integration.py integration_tests/
mv src/perception_nodes/test/test_nvdewarper_integration.py integration_tests/

echo "✅ Test reorganization complete!"
echo ""
echo "📋 Summary of changes:"
echo "  • Moved test_camera_pipeline_integration.py → integration_tests/"
echo "  • Moved test_nvdewarper_integration.py → integration_tests/"
echo "  • Unit tests remain in src/*/test/ directories"
echo ""
echo "🚀 Next steps:"
echo "  1. Update imports in moved test files"
echo "  2. Run tests to validate: ./scripts/test_reorganization_validate.sh"
echo "  3. Update CI configuration"
