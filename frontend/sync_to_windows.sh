#!/bin/bash
# Sync Flutter frontend from WSL to Windows for building

SOURCE_DIR="/home/lukehunker/ELEC49X_Group53_AIweightlift/frontend"
TARGET_DIR="/mnt/c/Users/lhunk/flutter_frontend"

echo "Syncing Flutter frontend to Windows..."

# Copy lib and assets (source files only)
rsync -av --delete "$SOURCE_DIR/lib/" "$TARGET_DIR/lib/"
rsync -av --delete "$SOURCE_DIR/assets/" "$TARGET_DIR/assets/" 2>/dev/null || true

# Copy config files
cp "$SOURCE_DIR/pubspec.yaml" "$TARGET_DIR/"
cp "$SOURCE_DIR/pubspec.lock" "$TARGET_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/analysis_options.yaml" "$TARGET_DIR/" 2>/dev/null || true

echo "✓ Sync complete!"
echo ""
echo "Next steps:"
echo "  1. cd $TARGET_DIR"
echo "  2. flutter pub get"
echo "  3. flutter run -d windows"
