#!/usr/bin/env bash
# okawhisp installer — Linux (systemd) + macOS (launchd)
# Usage: curl -sSL https://raw.githubusercontent.com/okahari/okawhisp/main/install.sh | bash
set -euo pipefail

REPO="https://raw.githubusercontent.com/okahari/okawhisp/main"
INSTALL_DIR="$HOME/.local/share/okawhisp"
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
    for cmd in xdotool pactl python3; do
        command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
    done

    if [ ${#MISSING[@]} -gt 0 ]; then
        info "Installing missing packages: ${MISSING[*]}"
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y xdotool pulseaudio-utils python3 \
                portaudio19-dev python3-dev 2>/dev/null \
                || warn "Auto-install failed. Run: sudo apt install xdotool portaudio19-dev python3-dev"
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y xdotool pipewire-utils python3 \
                portaudio-devel python3-devel 2>/dev/null \
                || warn "Auto-install failed. Run: sudo dnf install xdotool portaudio-devel python3-devel"
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm xdotool pipewire python portaudio 2>/dev/null \
                || warn "Auto-install failed. Run: sudo pacman -S xdotool pipewire python portaudio"
        elif command -v zypper &>/dev/null; then
            sudo zypper install -y xdotool pipewire-utils python3 \
                portaudio-devel 2>/dev/null \
                || warn "Auto-install failed."
        else
            warn "Unknown package manager. Install manually: ${MISSING[*]} + portaudio dev headers"
        fi
    fi

elif [ "$PLATFORM" = "macos" ]; then
    if ! command -v brew &>/dev/null; then
        warn "Homebrew not found. Install from https://brew.sh for automatic dependency management."
    else
        command -v python3 &>/dev/null || brew install python3
        brew list portaudio &>/dev/null 2>&1 || brew install portaudio
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
chmod +x "$SCRIPT"
curl -sSL "$REPO/sounds/start.mp3"   -o "$INSTALL_DIR/sounds/start.mp3"
curl -sSL "$REPO/sounds/stop.mp3"    -o "$INSTALL_DIR/sounds/stop.mp3"
ln -sf "$SCRIPT" "$BIN_DIR/okawhisp"
ok "Script installed to $INSTALL_DIR"

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
check_and_install "pyaudio"
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

if   [ "$VRAM_GB" -ge 8 ]; then RECOMMENDED="large-v3"; REASON="(8+ GB VRAM — best quality)"
elif [ "$VRAM_GB" -ge 6 ]; then RECOMMENDED="medium";   REASON="(6-8 GB VRAM — high quality)"
elif [ "$VRAM_GB" -ge 4 ]; then RECOMMENDED="small";    REASON="(4-6 GB VRAM — good quality)"
elif [ "$VRAM_GB" -ge 2 ]; then RECOMMENDED="base";     REASON="(2-4 GB VRAM — decent quality)"
else                             RECOMMENDED="tiny";     REASON="(CPU / low VRAM — fast)"
fi

echo ""
info "Whisper model selection:"
echo ""
echo "  tiny     — 75 MB,   very fast, decent quality"
echo "  base     — 145 MB,  fast,      good quality"
echo "  small    — 470 MB,  medium,    good quality"
echo "  medium   — 1.5 GB,  slower,    high quality"
echo "  large-v3 — 3 GB,    slowest,   best quality"
echo ""
echo "  Recommended for your system: ${RECOMMENDED} ${REASON}"
echo ""
# </dev/tty so read works even when script is piped via curl | bash
read -rp "  Choose model [${RECOMMENDED}]: " MODEL_CHOICE </dev/tty
MODEL_CHOICE="${MODEL_CHOICE:-$RECOMMENDED}"

case "$MODEL_CHOICE" in
    tiny)         WHISPER_MODEL="tiny" ;;
    base)         WHISPER_MODEL="base" ;;
    small)        WHISPER_MODEL="small" ;;
    medium)       WHISPER_MODEL="medium" ;;
    large|large-v3) WHISPER_MODEL="large-v3" ;;
    *)            WHISPER_MODEL="$RECOMMENDED"
                  # Re-apply mapping in case RECOMMENDED is a display name (e.g. "large")
                  [ "$WHISPER_MODEL" = "large" ] && WHISPER_MODEL="large-v3" ;;
esac

# Map model name to HuggingFace repo
case "$WHISPER_MODEL" in
    tiny)     HF_REPO="Systran/faster-whisper-tiny" ;;
    base)     HF_REPO="Systran/faster-whisper-base" ;;
    small)    HF_REPO="Systran/faster-whisper-small" ;;
    medium)   HF_REPO="Systran/faster-whisper-medium" ;;
    large-v3) HF_REPO="Systran/faster-whisper-large-v3" ;;
    *)        HF_REPO="Systran/faster-whisper-${WHISPER_MODEL}" ;;
esac

info "Selected: ${WHISPER_MODEL}"

# ── 6. Pre-download Whisper model ────────────────────────────────────────────
#
# Download the model NOW (before starting the service) so the service
# can start immediately without a download delay.
# Shows per-file progress so the user can see what is happening.
#
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
echo ""
info "Installing service..."

DISPLAY_VAL="${DISPLAY:-:0}"
XAUTH_VAL="${XAUTHORITY:-$HOME/.Xauthority}"
UID_VAL="$(id -u)"

if [ "$PLATFORM" = "linux" ]; then

    if ! command -v systemctl &>/dev/null; then
        warn "systemd not found."
        warn "Start manually: python3 $SCRIPT --key F9 --model $WHISPER_MODEL --engine faster"
    else
        SERVICE_DIR="$HOME/.config/systemd/user"
        SERVICE_FILE="$SERVICE_DIR/okawhisp.service"
        mkdir -p "$SERVICE_DIR"

        cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=OkaWhisp — System-Level Voice Input (F9 Hotkey)
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
        # Use local time (no -u) — journalctl --since expects local time
        START_TIME="$(date +'%Y-%m-%d %H:%M:%S')"

        if systemctl --user is-active --quiet okawhisp.service 2>/dev/null; then
            info "Restarting service with new config..."
            systemctl --user restart okawhisp.service
        else
            info "Starting service..."
            systemctl --user start okawhisp.service
        fi

        # Wait for ready signal — ONLY check logs written AFTER we started the service
        echo ""
        info "Waiting for service to become ready (model download may take several minutes)..."
        MAX_WAIT=600
        WAITED=0
        READY=0

        while [ "$WAITED" -lt "$MAX_WAIT" ]; do
            # Match "ready" signal in logs newer than our start timestamp
            if journalctl --user -u okawhisp.service \
                    --since "$START_TIME" --no-pager 2>/dev/null \
                    | grep -qiE "(hotkey|ready|listening|starte)"; then
                READY=1
                break
            fi

            # Detect early failure — no point waiting further
            if ! systemctl --user is-active --quiet okawhisp.service 2>/dev/null; then
                echo ""
                warn "Service exited unexpectedly. Recent logs:"
                echo ""
                journalctl --user -u okawhisp.service \
                    --since "$START_TIME" --no-pager -n 20 2>/dev/null || true
                echo ""
                warn "Fix the issue, then run: systemctl --user start okawhisp"
                break
            fi

            printf "."
            sleep 2
            WAITED=$((WAITED + 2))
        done

        echo ""
        if [ "$READY" -eq 1 ]; then
            ok "Service is ready!"
        elif systemctl --user is-active --quiet okawhisp.service 2>/dev/null; then
            warn "Service is running but did not signal ready within ${MAX_WAIT}s."
            warn "It may still be initializing. Check: journalctl --user -u okawhisp -f"
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
        <string>--key</string><string>F9</string>
        <string>--model</string><string>${WHISPER_MODEL}</string>
        <string>--engine</string><string>faster</string>
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
echo "  Press F9 to start recording."
echo ""
if [ "$PLATFORM" = "linux" ] && command -v systemctl &>/dev/null; then
    echo "  Logs:    journalctl --user -u okawhisp -f"
    echo "  Restart: systemctl --user restart okawhisp"
elif [ "$PLATFORM" = "macos" ]; then
    echo "  Logs:    tail -f ${INSTALL_DIR}/okawhisp.log"
    echo "  Restart: launchctl unload ${PLIST_FILE} && launchctl load ${PLIST_FILE}"
fi
echo "  Config:  ~/.config/okawhisp/config.toml"
echo ""
