#!/usr/bin/env bash
# okawhisp installer — Linux (systemd) + macOS (launchd)
# Usage: curl -sSL https://raw.githubusercontent.com/okahari/okawhisp/main/install.sh | bash
set -euo pipefail

REPO="https://raw.githubusercontent.com/okahari/okawhisp/main"
INSTALL_DIR="$HOME/.local/share/okawhisp"
CONFIG_DIR="$HOME/.config/okawhisp"
BIN_DIR="$HOME/.local/bin"
SCRIPT="$INSTALL_DIR/okawhisp.py"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
info() { echo -e "${YELLOW}→${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

echo ""
echo "  🎤  okawhisp installer"
echo "  ─────────────────────────────────────"
echo ""

# ── Detect OS ─────────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="macos" ;;
    *)      err "Unsupported OS: $OS. Linux and macOS are supported." ;;
esac

# ── 1. System dependencies ───────────────────────────────────────────────────
info "Checking system dependencies..."

if [ "$PLATFORM" = "linux" ]; then
    MISSING=()
    for cmd in xdotool pactl parec python3; do
        command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
    done

    if [ ${#MISSING[@]} -gt 0 ]; then
        info "Installing missing packages: ${MISSING[*]}"
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y xdotool pulseaudio-utils python3 2>/dev/null \
                || warn "Auto-install failed. Run: sudo apt install xdotool pulseaudio-utils"
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y xdotool pulseaudio-utils python3 2>/dev/null \
                || warn "Auto-install failed. Run: sudo dnf install xdotool pulseaudio-utils"
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm xdotool libpulse python 2>/dev/null \
                || warn "Auto-install failed. Run: sudo pacman -S xdotool libpulse python"
        elif command -v zypper &>/dev/null; then
            sudo zypper install -y xdotool pulseaudio-utils python3 2>/dev/null \
                || warn "Auto-install failed."
        else
            warn "Unknown package manager. Install manually: ${MISSING[*]}"
        fi
    fi

elif [ "$PLATFORM" = "macos" ]; then
    if ! command -v brew &>/dev/null; then
        warn "Homebrew not found. Install from https://brew.sh for automatic dependency management."
    else
        command -v python3 &>/dev/null || brew install python3
        brew list pulseaudio &>/dev/null 2>&1 || brew install pulseaudio
    fi
fi

ok "System dependencies OK"

# ── 2. pip ───────────────────────────────────────────────────────────────────
info "Checking pip..."
if ! python3 -m pip --version &>/dev/null; then
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y python3-pip 2>/dev/null \
            || err "Could not install pip. Please install python3-pip manually."
    else
        err "pip not found. Install python3-pip for your distribution."
    fi
fi
ok "pip ready"

# ── 3. Download script + assets ──────────────────────────────────────────────
info "Installing okawhisp script..."
mkdir -p "$INSTALL_DIR" "$BIN_DIR" "$INSTALL_DIR/sounds"
curl -sSL "$REPO/okawhisp.py"        -o "$SCRIPT"
curl -sSL "$REPO/okawhispctl.py"     -o "$INSTALL_DIR/okawhispctl.py"
chmod +x "$SCRIPT" "$INSTALL_DIR/okawhispctl.py"
curl -sSL "$REPO/sounds/start.mp3"   -o "$INSTALL_DIR/sounds/start.mp3"
curl -sSL "$REPO/sounds/stop.mp3"    -o "$INSTALL_DIR/sounds/stop.mp3"
ln -sf "$INSTALL_DIR/okawhispctl.py" "$BIN_DIR/okawhisp"
ok "Script installed to $INSTALL_DIR"

# ── 3b. Default config (only if none exists) ─────────────────────────────────
if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    info "Creating default config..."
    mkdir -p "$CONFIG_DIR"
    curl -sSL "$REPO/config.example.toml" -o "$CONFIG_DIR/config.toml"
    ok "Config created at $CONFIG_DIR/config.toml"
else
    ok "Config exists at $CONFIG_DIR/config.toml (preserved)"
fi

# ── 4. Python dependencies ───────────────────────────────────────────────────
echo ""
info "Checking Python dependencies..."
echo ""

check_and_install() {
    local pkg=$1 import_name=${2:-$1}
    echo -n "  Checking $pkg... "
    if python3 -c "import $import_name" 2>/dev/null; then
        echo -e "${GREEN}installed${NC}"; return 0
    fi
    echo -e "${YELLOW}missing${NC}"
    echo -n "  Installing $pkg... "
    if python3 -m pip install --user --break-system-packages "$pkg" >/dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"; return 0
    fi
    echo -e "${RED}FAILED${NC}"
    echo "  Retrying with output:"
    python3 -m pip install --user --break-system-packages "$pkg"
}

check_and_install "torch"
check_and_install "numpy"
check_and_install "faster-whisper"     "faster_whisper"
check_and_install "silero-vad"         "silero_vad"
check_and_install "pynput"
check_and_install "huggingface_hub"    "huggingface_hub"
check_and_install "hf-transfer"        "hf_transfer"

echo ""
ok "Python dependencies ready"

# ── 5. Model selection (GPU-aware) ───────────────────────────────────────────
echo ""
info "Detecting GPU capabilities..."

VRAM_GB=0
if command -v nvidia-smi &>/dev/null; then
    VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" -gt 0 ] 2>/dev/null; then
        VRAM_GB=$((VRAM_MB / 1024))
        echo "  GPU detected: ${VRAM_GB} GB VRAM"
    fi
fi

if   [ "$VRAM_GB" -ge 8 ]; then WHISPER_MODEL="large-v3"; REASON="8+ GB VRAM"
elif [ "$VRAM_GB" -ge 6 ]; then WHISPER_MODEL="medium";   REASON="6-8 GB VRAM"
elif [ "$VRAM_GB" -ge 4 ]; then WHISPER_MODEL="small";    REASON="4-6 GB VRAM"
elif [ "$VRAM_GB" -ge 2 ]; then WHISPER_MODEL="base";     REASON="2-4 GB VRAM"
else                             WHISPER_MODEL="tiny";     REASON="CPU / low VRAM"
fi

info "Model: ${WHISPER_MODEL} (${REASON})"

case "$WHISPER_MODEL" in
    tiny)     HF_REPO="Systran/faster-whisper-tiny" ;;
    base)     HF_REPO="Systran/faster-whisper-base" ;;
    small)    HF_REPO="Systran/faster-whisper-small" ;;
    medium)   HF_REPO="Systran/faster-whisper-medium" ;;
    large-v3) HF_REPO="Systran/faster-whisper-large-v3" ;;
    *)        HF_REPO="Systran/faster-whisper-${WHISPER_MODEL}" ;;
esac

# ── 6. Pre-download Whisper model ────────────────────────────────────────────
echo ""
info "Downloading Whisper model '${WHISPER_MODEL}' (once — cached for future starts)..."
echo ""

python3 - <<PYEOF
import sys, os
from pathlib import Path

# ── Fast parallel download via hf_transfer (Rust, multi-thread) ──────────────
_HF_FAST = False
try:
    import hf_transfer as _  # noqa – just check it's importable
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    _HF_FAST = True
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

# ── Force tqdm to stdout, always enabled (patch BEFORE importing huggingface_hub)
import tqdm as _tqdm
import tqdm.auto as _tqdm_auto

class _Tqdm(_tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = False
        kwargs["file"]    = sys.stdout
        super().__init__(*args, **kwargs)

_tqdm.tqdm      = _Tqdm
_tqdm_auto.tqdm = _Tqdm

# ── Now import huggingface_hub (will use our patched tqdm) ───────────────────
from huggingface_hub import repo_info, hf_hub_download, try_to_load_from_cache

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
NC     = "\033[0m"

def fmt_size(n):
    if not n: return "?"
    if n >= 1 << 30: return f"{n / (1<<30):.1f} GB"
    if n >= 1 << 20: return f"{n / (1<<20):.0f} MB"
    if n >= 1 << 10: return f"{n / (1<<10):.0f} KB"
    return f"{n} B"

try:
    repo_id = "${HF_REPO}"
    mode    = f"{'hf-transfer (parallel)' if _HF_FAST else 'standard'}"
    sys.stdout.write(f"  Fetching file list from HuggingFace... [{mode}]\n")
    sys.stdout.flush()

    meta  = repo_info(repo_id, files_metadata=True)
    files = sorted(
        [(s.rfilename, getattr(s, "size", None) or 0) for s in meta.siblings],
        key=lambda x: x[1], reverse=True
    )

    to_dl    = []
    n_cached = 0
    for fname, size in files:
        cached = try_to_load_from_cache(repo_id, fname)
        if cached and Path(cached).exists():
            n_cached += 1
        else:
            to_dl.append((fname, size))

    total_size = sum(s for _, s in files)

    if not to_dl:
        sys.stdout.write(
            f"  {GREEN}All {len(files)} files already cached{NC}"
            f" ({fmt_size(total_size)} total)\n\n"
        )
        sys.stdout.flush()
    else:
        dl_size  = sum(s for _, s in to_dl)
        cached_s = f", {n_cached} already cached" if n_cached else ""
        sys.stdout.write(
            f"  {len(to_dl)} file(s) to download"
            f" ({fmt_size(dl_size)}{cached_s})\n\n"
        )
        sys.stdout.flush()

        for i, (fname, size) in enumerate(to_dl, 1):
            sys.stdout.write(f"  [{i}/{len(to_dl)}] {fname}  ({fmt_size(size)})\n")
            sys.stdout.flush()
            hf_hub_download(repo_id=repo_id, filename=fname)
            sys.stdout.write(f"  [{i}/{len(to_dl)}] {GREEN}done{NC}  {fname}\n\n")
            sys.stdout.flush()

except Exception as exc:
    sys.stdout.write(f"\n  {YELLOW}Warning:{NC} {exc}\n")
    sys.stdout.write("  Model will be downloaded when the service first starts.\n\n")
    sys.stdout.flush()
PYEOF

ok "Model ready"

# ── 7. Install and start service ─────────────────────────────────────────────
# All settings come from config.toml — no CLI args in the service file.
# This means changes to config.toml take effect on service restart without
# needing to re-run the installer.
echo ""
info "Installing user service (systemctl --user)..."

DISPLAY_VAL="${DISPLAY:-:0}"
XAUTH_VAL="${XAUTHORITY:-$HOME/.Xauthority}"
UID_VAL="$(id -u)"

if [ "$PLATFORM" = "linux" ]; then

    if ! command -v systemctl &>/dev/null; then
        warn "systemd not found."
        warn "Start manually: python3 $SCRIPT"
    else
        SERVICE_DIR="$HOME/.config/systemd/user"
        SERVICE_FILE="$SERVICE_DIR/okawhisp.service"
        mkdir -p "$SERVICE_DIR"

        cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=OkaWhisp — System-Level Voice Input
After=graphical-session.target pipewire.service
StartLimitBurst=3
StartLimitIntervalSec=60s

[Service]
Type=simple
ExecStart=/usr/bin/python3 ${SCRIPT}
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
        systemctl --user enable okawhisp.service

        # Timestamp BEFORE starting — journal polling will only look at entries after this
        START_TIME="$(date +'%Y-%m-%d %H:%M:%S')"

        if systemctl --user is-active --quiet okawhisp.service 2>/dev/null; then
            systemctl --user restart okawhisp.service
        else
            systemctl --user start okawhisp.service
        fi

        # Display: stream filtered journal output in background (visual only)
        # Detection: poll journal snapshots in foreground (avoids journalctl -f deadlock)
        echo ""
        info "Waiting for model to load..."
        echo ""

        ( journalctl --user -u okawhisp.service \
              --since "$START_TIME" -f -o cat 2>/dev/null | \
          grep --line-buffered -vE \
              "^(Stopping|Stopped|Started|Failed|okawhisp\.service:)|={5,}|OkaWhisp (finished|starting|- System)|Log file:|Progress may not|^[[:space:]]*[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | \
          while IFS= read -r line; do
              [ -n "$line" ] && printf "  %s\n" "$line"
          done ) &
        STREAM_PID=$!

        MAX_WAIT=600
        WAITED=0
        READY=0
        while [ "$WAITED" -lt "$MAX_WAIT" ]; do
            if ! systemctl --user is-active --quiet okawhisp.service 2>/dev/null; then
                break
            fi
            if journalctl --user -u okawhisp.service \
                    --since "$START_TIME" --no-pager -o cat 2>/dev/null | \
               grep -qiE "(Starting hotkey|🎹|Hotkey listener|Audio initialized|parec initialized|input stream)"; then
                READY=1
                break
            fi
            sleep 2
            WAITED=$((WAITED + 2))
        done

        kill "$STREAM_PID" 2>/dev/null
        wait "$STREAM_PID" 2>/dev/null
        echo ""

        # Read hotkey from config for display
        HOTKEY="ALT_GR"
        if [ -f "$CONFIG_DIR/config.toml" ]; then
            CFG_KEY="$(grep -E '^\s*key\s*=' "$CONFIG_DIR/config.toml" 2>/dev/null | head -1 | sed 's/.*=\s*"\?\([^"]*\)"\?.*/\1/' || true)"
            [ -n "$CFG_KEY" ] && HOTKEY="$CFG_KEY"
        fi

        if [ "$READY" -eq 1 ]; then
            ok "Service ready! Press ${HOTKEY} to record."
        elif ! systemctl --user is-active --quiet okawhisp.service 2>/dev/null; then
            warn "Service stopped. Check: journalctl --user -u okawhisp -f"
        else
            warn "Timeout. Check: journalctl --user -u okawhisp -f"
        fi
    fi

elif [ "$PLATFORM" = "macos" ]; then

    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="$PLIST_DIR/ai.okahari.okawhisp.plist"
    mkdir -p "$PLIST_DIR"

    cat > "$PLIST_FILE" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.okahari.okawhisp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>${SCRIPT}</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key>
    <string>${INSTALL_DIR}/okawhisp.log</string>
    <key>StandardErrorPath</key>
    <string>${INSTALL_DIR}/okawhisp-error.log</string>
</dict>
</plist>
EOF

    launchctl unload "$PLIST_FILE" 2>/dev/null || true
    launchctl load  "$PLIST_FILE"
    ok "launchd service installed and started"

fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}  ✓ okawhisp installed and ready!${NC}"
echo ""
if [ "$PLATFORM" = "linux" ] && command -v systemctl &>/dev/null; then
    echo "  Logs:    journalctl --user -u okawhisp -f"
    echo "  Status:  systemctl --user status okawhisp"
    echo "  Restart: systemctl --user restart okawhisp"
elif [ "$PLATFORM" = "macos" ]; then
    echo "  Logs:    tail -f ${INSTALL_DIR}/okawhisp.log"
    echo "  Restart: launchctl unload ${PLIST_FILE} && launchctl load ${PLIST_FILE}"
fi
echo "  Config:  $CONFIG_DIR/config.toml"
echo ""
