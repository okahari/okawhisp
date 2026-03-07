#!/usr/bin/env bash
# okawhisp installer
# Usage: curl -sSL https://raw.githubusercontent.com/okahari/okawhisp/main/install.sh | bash
set -euo pipefail

REPO="https://raw.githubusercontent.com/okahari/okawhisp/main"
INSTALL_DIR="$HOME/.local/share/okawhisp"
BIN_DIR="$HOME/.local/bin"
SERVICE_DIR="$HOME/.config/systemd/user"
SCRIPT="$INSTALL_DIR/okawhisp.py"
SERVICE="$SERVICE_DIR/okawhisp.service"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
info() { echo -e "${YELLOW}→${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

echo ""
echo "  🎤  okawhisp installer"
echo "  ─────────────────────────────────────"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────────────
info "Checking system dependencies..."
MISSING=()
for cmd in xdotool pactl python3; do
    command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    info "Installing missing system packages: ${MISSING[*]}"
    # Map commands to packages
    declare -A PKG_MAP=(
        [xdotool]="xdotool"
        [pactl]="pipewire-audio"
        [python3]="python3"
    )
    PKGS=()
    for cmd in "${MISSING[@]}"; do
        PKGS+=("${PKG_MAP[$cmd]:-$cmd}")
    done
    sudo apt-get install -y "${PKGS[@]}" portaudio19-dev python3-dev 2>/dev/null \
        || info "Could not auto-install packages. Please run: sudo apt install xdotool portaudio19-dev python3-dev"
fi
ok "System dependencies OK"

# ── 2. pip ────────────────────────────────────────────────────────────────────
if ! python3 -m pip --version &>/dev/null; then
    info "Installing pip..."
    sudo apt-get install -y python3-pip 2>/dev/null \
        || err "Could not install pip. Please install python3-pip manually."
fi
ok "pip ready"

# ── 3. Download script ────────────────────────────────────────────────────────
info "Installing okawhisp script..."
mkdir -p "$INSTALL_DIR" "$BIN_DIR" "$INSTALL_DIR/sounds"
curl -sSL "$REPO/okawhisp.py" -o "$SCRIPT"
chmod +x "$SCRIPT"

# Download sound files
curl -sSL "$REPO/sounds/start.mp3" -o "$INSTALL_DIR/sounds/start.mp3"
curl -sSL "$REPO/sounds/stop.mp3" -o "$INSTALL_DIR/sounds/stop.mp3"

# Convenience symlink
ln -sf "$SCRIPT" "$BIN_DIR/okawhisp"
ok "Script installed to $INSTALL_DIR"

# ── 4. Check Python dependencies ──────────────────────────────────────────────
info "Checking Python dependencies..."
MISSING_PY=()

python3 -c "import torch" 2>/dev/null || MISSING_PY+=("torch")
python3 -c "import faster_whisper" 2>/dev/null || MISSING_PY+=("faster-whisper")
python3 -c "import numpy" 2>/dev/null || MISSING_PY+=("numpy")
python3 -c "import pyaudio" 2>/dev/null || MISSING_PY+=("pyaudio")
python3 -c "import pynput" 2>/dev/null || MISSING_PY+=("pynput")

python3 -c "import silero_vad" 2>/dev/null || MISSING_PY+=("silero-vad")

if [ ${#MISSING_PY[@]} -eq 0 ]; then
    ok "All Python dependencies already installed"
else
    info "Installing missing Python packages: ${MISSING_PY[*]}"
    info "This may take 1-5 min on first install (torch+CUDA ~2GB)"
    
    # Install via pip --user (uses system site-packages, no isolated env)
    for pkg in "${MISSING_PY[@]}"; do
        python3 -m pip install --user --break-system-packages "$pkg" --quiet 2>/dev/null \
            || info "Could not install $pkg - will be installed on first launch"
    done
    ok "Python dependencies installed"
fi

# ── 5. Systemd service ────────────────────────────────────────────────────────
info "Installing systemd service..."
mkdir -p "$SERVICE_DIR"

# Detect display
DISPLAY_VAL="${DISPLAY:-:0}"
XAUTH_VAL="${XAUTHORITY:-$HOME/.Xauthority}"
UID_VAL="$(id -u)"

cat > "$SERVICE" <<EOF
[Unit]
Description=OkaWhisp - System-Level Voice Input (F9 Hotkey)
After=graphical-session.target pipewire.service
StartLimitBurst=3
StartLimitIntervalSec=60s

[Service]
Type=simple
ExecStart=/usr/bin/python3 ${SCRIPT} --key F9 --model large-v3 --engine faster --language de --beam-size 5 --silence 2.0

Environment="DISPLAY=${DISPLAY_VAL}"
Environment="XAUTHORITY=${XAUTH_VAL}"
Environment="XDG_RUNTIME_DIR=/run/user/${UID_VAL}"
Environment="PYTHONUNBUFFERED=1"
Environment="CUDA_VISIBLE_DEVICES=0"

Restart=always
RestartSec=5s
MemoryMax=4G
MemoryHigh=3G

StandardOutput=journal
StandardError=journal
SyslogIdentifier=okawhisp

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now okawhisp.service
ok "Service installed and started"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}  ✓ okawhisp installed!${NC}"
echo ""
echo "  Press F9 to start recording."
echo "  First start downloads the Whisper model (~1-3 GB) — be patient."
echo ""
echo "  Logs:    journalctl --user -u okawhisp -f"
echo "  Restart: systemctl --user restart okawhisp"
echo "  Config:  ~/.config/okawhisp/config.toml"
echo ""
