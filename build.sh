#!/bin/bash

set -e

APP_NAME=$(grep -m 1 '^name =' Cargo.toml | cut -d '"' -f 2)
VERSION=$(grep -m 1 '^version =' Cargo.toml | cut -d '"' -f 2)

echo "Building $APP_NAME v$VERSION"

BUILD_DIR="dist"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Building Linux Release..."
cargo build --release

echo "Building Windows Release..."
rustup target add x86_64-pc-windows-gnu >/dev/null 2>&1
cargo build --release --target x86_64-pc-windows-gnu

echo "Packaging Linux..."
LINUX_PKG="${APP_NAME}-v${VERSION}-linux-x86_64"
mkdir -p "$BUILD_DIR/$LINUX_PKG"
cp target/release/"$APP_NAME" "$BUILD_DIR/$LINUX_PKG/"
cp -r presets "$BUILD_DIR/$LINUX_PKG/"
cp -r static "$BUILD_DIR/$LINUX_PKG/"
cd "$BUILD_DIR" && zip -r "${LINUX_PKG}.zip" "$LINUX_PKG" && cd ..

echo "Packaging Windows..."
WIN_PKG="${APP_NAME}-v${VERSION}-windows-x86_64"
mkdir -p "$BUILD_DIR/$WIN_PKG"
cp target/x86_64-pc-windows-gnu/release/"$APP_NAME".exe "$BUILD_DIR/$WIN_PKG/"
cp -r presets "$BUILD_DIR/$WIN_PKG/"
cp -r static "$BUILD_DIR/$WIN_PKG/"
cd "$BUILD_DIR" && zip -r "${WIN_PKG}.zip" "$WIN_PKG" && cd ..

echo "Done, zips are in the /$BUILD_DIR/ folder"
ls -lh "$BUILD_DIR"/*.zip
