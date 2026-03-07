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
info "Checking pip..."
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

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo ""
info "Checking Python dependencies..."
echo ""

check_and_install() {
    local pkg=$1
    local import_name=${2:-$pkg}
    
    echo -n "  Checking $pkg... "
    if python3 -c "import $import_name" 2>/dev/null; then
        echo -e "${GREEN}installed${NC}"
        return 0
    else
        echo -e "${YELLOW}missing${NC}"
        echo -n "  Installing $pkg... "
        
        if python3 -m pip install --user --break-system-packages "$pkg" >/dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
            return 0
        else
            echo -e "${RED}FAILED${NC}"
            echo ""
            echo "  Trying to install $pkg with verbose output:"
            python3 -m pip install --user --break-system-packages "$pkg"
            return $?
        fi
    fi
}

check_and_install "torch"
check_and_install "numpy"
check_and_install "pyaudio"
check_and_install "faster-whisper" "faster_whisper"
check_and_install "silero-vad" "silero_vad"
check_and_install "pynput"

# Install hf_transfer for 3-5x faster HuggingFace downloads
if ! python3 -c "import hf_transfer" 2>/dev/null; then
    echo -n "  Installing hf_transfer for faster downloads... "
    if pip install --user --break-system-packages hf_transfer >/dev/null 2>&1; then
        echo -e "${GREEN}done${NC}"
    else
        echo -e "${YELLOW}skipped (optional)${NC}"
    fi
fi

echo ""
ok "Python dependencies ready"

# ── 5. Model selection (GPU-aware) ───────────────────────────────────────────
echo ""
info "Detecting GPU capabilities..."

# Detect VRAM
VRAM_GB=0
if command -v nvidia-smi &>/dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$VRAM_MB" ]; then
        VRAM_GB=$((VRAM_MB / 1024))
        echo "  GPU found: ${VRAM_GB} GB VRAM"
    fi
fi

# Recommend model based on VRAM (auto-detection)
if [ $VRAM_GB -ge 8 ]; then
    RECOMMENDED="large"
    REASON="(8+ GB VRAM - best quality)"
elif [ $VRAM_GB -ge 6 ]; then
    RECOMMENDED="medium"
    REASON="(6-8 GB VRAM - high quality)"
elif [ $VRAM_GB -ge 4 ]; then
    RECOMMENDED="small"
    REASON="(4-6 GB VRAM - good quality)"
elif [ $VRAM_GB -ge 2 ]; then
    RECOMMENDED="base"
    REASON="(2-4 GB VRAM - decent quality)"
else
    RECOMMENDED="tiny"
    REASON="(CPU or low VRAM - fast)"
fi

echo ""
info "Whisper model selection:"
echo ""
echo "  tiny   - 75 MB download, very fast, decent quality"
echo "  base   - 145 MB download, fast, good quality"
echo "  small  - 470 MB download, balanced, good quality"
echo "  medium - 1.5 GB download, slower, high quality"
echo "  large  - 3 GB download, slowest, best quality"
echo ""
echo "  Recommended for your system: ${RECOMMENDED} ${REASON}"
echo ""
read -p "Choose model [${RECOMMENDED}]: " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-$RECOMMENDED}

# Map to actual model names
case "$MODEL_CHOICE" in
    tiny)   WHISPER_MODEL="tiny" ;;
    base)   WHISPER_MODEL="base" ;;
    small)  WHISPER_MODEL="small" ;;
    medium) WHISPER_MODEL="medium" ;;
    large)  WHISPER_MODEL="large-v3" ;;
    *)      WHISPER_MODEL="$RECOMMENDED" ;;
esac

info "Selected: ${WHISPER_MODEL}"

# ── 6. Systemd service ────────────────────────────────────────────────────────
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
ExecStart=/usr/bin/python3 ${SCRIPT} --key F9 --model ${WHISPER_MODEL} --engine faster --language de --beam-size 5 --silence 2.0

Environment="DISPLAY=${DISPLAY_VAL}"
Environment="XAUTHORITY=${XAUTH_VAL}"
Environment="XDG_RUNTIME_DIR=/run/user/${UID_VAL}"
Environment="PYTHONUNBUFFERED=1"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="HF_HUB_ENABLE_HF_TRANSFER=1"

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
systemctl --user enable okawhisp.service

# ── 6. Download model BEFORE starting service ────────────────────────────────
echo ""

# Model sizes
case "$WHISPER_MODEL" in
    tiny)     MODEL_MB=75 ;;
    base)     MODEL_MB=145 ;;
    small)    MODEL_MB=470 ;;
    medium)   MODEL_MB=1500 ;;
    large-v3) MODEL_MB=3000 ;;
    large)    MODEL_MB=3000 ;;
    *)        MODEL_MB=1000 ;;
esac

info "Downloading Whisper model '${WHISPER_MODEL}' (~${MODEL_MB} MB)..."
echo ""

# Map model name to repo name for cache dir
case "$WHISPER_MODEL" in
    large) MODEL_REPO_NAME="large-v3" ;;
    *) MODEL_REPO_NAME="$WHISPER_MODEL" ;;
esac

MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-${MODEL_REPO_NAME}"

# Python download script runs in background
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 << 'DOWNLOAD_SCRIPT' &
import sys, os
from huggingface_hub import snapshot_download

model_name = os.environ.get("WHISPER_MODEL", "small")
repo_map = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
    "large-v3": "Systran/faster-whisper-large-v3",
}
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
try:
    snapshot_download(repo_id=repo_map.get(model_name, repo_map["small"]), cache_dir=cache_dir)
    sys.exit(0)
except Exception as e:
    print(f"Download failed: {e}", file=sys.stderr)
    sys.exit(1)
DOWNLOAD_SCRIPT

DOWNLOAD_PID=$!
export WHISPER_MODEL

# Monitor download progress
SPINNER="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPIN_IDX=0

while kill -0 $DOWNLOAD_PID 2>/dev/null; do
    # Check cache size
    if [ -d "$MODEL_CACHE_DIR" ]; then
        DOWNLOADED_KB=$(du -sk "$MODEL_CACHE_DIR" 2>/dev/null | cut -f1)
        DOWNLOADED_MB=$((DOWNLOADED_KB / 1024))
        PROGRESS=$((DOWNLOADED_MB * 100 / MODEL_MB))
        [ $PROGRESS -gt 99 ] && PROGRESS=99
    else
        DOWNLOADED_MB=0
        PROGRESS=0
    fi
    
    # Draw progress bar
    FILLED=$((PROGRESS / 5))
    EMPTY=$((20 - FILLED))
    BAR=$(printf "${GREEN}▓%.0s${NC}" $(seq 1 $FILLED))$(printf "░%.0s" $(seq 1 $EMPTY))
    
    SPIN_CHAR=$(echo "$SPINNER" | cut -c$((SPIN_IDX + 1)))
    SPIN_IDX=$(( (SPIN_IDX + 1) % 10 ))
    
    echo -ne "\r  $SPIN_CHAR $BAR ${PROGRESS}% (${DOWNLOADED_MB}/${MODEL_MB} MB)"
    
    sleep 2
done

# Wait for download to finish
wait $DOWNLOAD_PID
DOWNLOAD_EXIT=$?

echo -ne "\r\033[K"
if [ $DOWNLOAD_EXIT -ne 0 ]; then
    err "Model download failed"
fi

echo -e "  ${GREEN}▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓${NC} 100%"
ok "Model downloaded to cache"

# ── 7. Now start service (model loads from cache instantly) ──────────────────
echo ""
if systemctl --user is-active --quiet okawhisp.service; then
    info "Restarting service..."
    systemctl --user restart okawhisp.service
else
    info "Starting service..."
    systemctl --user start okawhisp.service
fi

# Wait for service ready (should be fast - model is in cache)
echo -n "  Waiting for service"
MAX_WAIT=60
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if journalctl --user -u okawhisp.service --since "1 minute ago" --no-pager 2>/dev/null | grep -qE "(Starte Hotkey-Listener|🎹 Hotkey: F9)"; then
        echo -e "\r\033[K"
        ok "Service ready!"
        break
    fi
    
    if ! systemctl --user is-active --quiet okawhisp.service; then
        echo -e "\r\033[K"
        err "Service crashed! Check: journalctl --user -u okawhisp -n 30"
    fi
    
    echo -ne "\r  Waiting for service... ${WAITED}s"
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "\r\033[K"
    warn "Service did not become ready. Check: journalctl --user -u okawhisp -f"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}  ✓ okawhisp installed and ready!${NC}"
echo ""
echo "  Press F9 to start recording."
echo ""
echo "  Logs:    journalctl --user -u okawhisp -f"
echo "  Restart: systemctl --user restart okawhisp"
echo "  Config:  ~/.config/okawhisp/config.toml"
echo ""
# Updated Sa 07 Mär 2026 06:34:01 CET
