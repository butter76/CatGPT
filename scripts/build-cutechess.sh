#!/bin/bash
# Build cutechess-cli from source.
#
# cutechess-cli is the engine-vs-engine match runner used by the web/
# tournament environment. Modern cutechess builds with CMake; we configure the
# project and build only the `cli` target (the GUI is never linked), though
# CMake still requires the full Qt component set at configure time.
#
# Usage:
#   ./scripts/build-cutechess.sh [install_prefix]
#
# install_prefix defaults to "$HOME/cutechess". The built binary is copied to
#   <install_prefix>/cutechess-cli
# Point the web app at it via CUTECHESS_CLI_PATH in web/.env.local.

set -euo pipefail

PREFIX="${1:-$HOME/cutechess}"
SRC_DIR="$PREFIX/src"
BUILD_DIR="$SRC_DIR/build"
REPO_URL="https://github.com/cutechess/cutechess.git"

log() { printf '\033[1;36m[build-cutechess]\033[0m %s\n' "$*"; }

# ── 1. System dependencies (Qt5 + CMake + build tools) ───────────────
# CMake's find_package requires Core/Gui/Widgets/Concurrent/Svg/PrintSupport
# even when only the CLI target is built.
if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
    log "Installing Qt5 + CMake build dependencies via apt-get..."
    $SUDO apt-get update -y
    $SUDO apt-get install -y --no-install-recommends \
        build-essential git cmake \
        qtbase5-dev qtbase5-dev-tools libqt5svg5-dev
else
    log "apt-get not found — ensure cmake, a C++ toolchain, and Qt5 dev"
    log "packages (qtbase5-dev + libqt5svg5-dev) are installed."
fi

# ── 2. Clone (or update) the source ──────────────────────────────────
mkdir -p "$PREFIX"
if [ -d "$SRC_DIR/.git" ]; then
    log "Updating existing checkout at $SRC_DIR"
    git -C "$SRC_DIR" fetch --depth 1 origin
    git -C "$SRC_DIR" reset --hard origin/HEAD
else
    log "Cloning cutechess into $SRC_DIR"
    git clone --depth 1 "$REPO_URL" "$SRC_DIR"
fi

JOBS="$(nproc 2>/dev/null || echo 4)"

# ── 3. Configure with CMake and build only the CLI target ────────────
log "Configuring (CMake)..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_TESTS=OFF

log "Building cutechess-cli (target: cli)..."
cmake --build "$BUILD_DIR" --target cli -j"$JOBS"

CLI_BIN="$BUILD_DIR/cutechess-cli"
if [ ! -x "$CLI_BIN" ]; then
    echo "ERROR: build finished but $CLI_BIN was not produced." >&2
    exit 1
fi

cp "$CLI_BIN" "$PREFIX/cutechess-cli"

log "Done."
log "Binary: $PREFIX/cutechess-cli"
"$PREFIX/cutechess-cli" --version || true
echo
log "Add this to web/.env.local:"
echo "    CUTECHESS_CLI_PATH=$PREFIX/cutechess-cli"
