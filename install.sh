#!/usr/bin/env bash
# voice-type installer
# Usage: curl -sSL https://raw.githubusercontent.com/YOUR_USER/voice-type/main/install.sh | bash
set -euo pipefail

REPO="https://raw.githubusercontent.com/YOUR_USER/voice-type/main"
INSTALL_DIR="$HOME/.local/share/voice-type"
BIN_DIR="$HOME/.local/bin"
SERVICE_DIR="$HOME/.config/systemd/user"
SCRIPT="$INSTALL_DIR/voice-type.py"
SERVICE="$SERVICE_DIR/voice-type.service"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
info() { echo -e "${YELLOW}→${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

echo ""
echo "  🎤  voice-type installer"
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

# ── 2. uv ─────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
    info "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
UV="${HOME}/.local/bin/uv"
[ -f "$UV" ] || UV="$(command -v uv)"
ok "uv ready: $UV"

# ── 3. Download script ────────────────────────────────────────────────────────
info "Installing voice-type script..."
mkdir -p "$INSTALL_DIR" "$BIN_DIR"
curl -sSL "$REPO/voice-type.py" -o "$SCRIPT"
chmod +x "$SCRIPT"

# Convenience symlink
ln -sf "$SCRIPT" "$BIN_DIR/voice-type"
ok "Script installed to $SCRIPT"

# ── 4. Pre-warm dependencies (runs in background, first launch may be slow otherwise) ──
info "Pre-installing Python dependencies (this takes ~30s on first run)..."
"$UV" run --with faster-whisper --with silero-vad --with pyaudio --with numpy \
    python3 -c "import faster_whisper, numpy, pyaudio; print('deps OK')" 2>/dev/null \
    || info "Deps will be installed on first launch"

# ── 5. Systemd service ────────────────────────────────────────────────────────
info "Installing systemd service..."
mkdir -p "$SERVICE_DIR"

# Detect display
DISPLAY_VAL="${DISPLAY:-:0}"
XAUTH_VAL="${XAUTHORITY:-$HOME/.Xauthority}"
UID_VAL="$(id -u)"

cat > "$SERVICE" <<EOF
[Unit]
Description=Voice Type - System-Level Voice Input (F9 Hotkey)
After=graphical-session.target pipewire.service
StartLimitBurst=3
StartLimitIntervalSec=60s

[Service]
Type=simple
ExecStart=${UV} run ${SCRIPT} --key F9 --model large-v3 --engine faster --language de --beam-size 5 --silence 2.0

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
SyslogIdentifier=voice-type

[Install]
WantedBy=graphical-session.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now voice-type.service
ok "Service installed and started"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}  ✓ voice-type installed!${NC}"
echo ""
echo "  Press F9 to start recording."
echo "  First start downloads the Whisper model (~1-3 GB) — be patient."
echo ""
echo "  Logs:    journalctl --user -u voice-type -f"
echo "  Restart: systemctl --user restart voice-type"
echo "  Config:  ~/.config/voice-type/config.toml"
echo ""
