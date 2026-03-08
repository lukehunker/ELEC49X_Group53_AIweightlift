#!/bin/bash

# Flutter build script for Vercel

echo "Downloading Flutter SDK..."
git clone https://github.com/flutter/flutter.git -b stable
export PATH="$PATH:`pwd`/flutter/bin"

echo "Flutter version:"
flutter --version

echo "Getting dependencies..."
flutter pub get

echo "Building Flutter web app..."
# Pass the Vercel API_BASE_URL environment variable to the Dart compiler
if [ -z "$API_BASE_URL" ]; then
    echo "API_BASE_URL is not set. Using default."
    flutter build web --release
else
    echo "Building with API_BASE_URL=$API_BASE_URL"
    flutter build web --release --dart-define=API_BASE_URL=$API_BASE_URL
fi
