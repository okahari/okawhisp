# OkaWhisp

**Local voice typing for Linux** — speak and your words appear anywhere.

Press a hotkey, speak, text is typed into the focused window. Terminal, browser, chat, editor — anything.

## Install

```bash
curl -sSL https://github.com/okahari/okawhisp/raw/main/install.sh | bash
```

Installs all dependencies (system packages + Python), downloads the Whisper model, and starts the service. Works on Debian/Ubuntu, Fedora, Arch, and openSUSE.

After install, press **AltGr** (default hotkey) and start talking.

## Features

- **Global hotkey** (AltGr default, F1-F12 configurable) — works in any window
- **Push-to-Talk + Toggle mode** — hold key to talk, or tap to start/stop
- **Smart silence detection** via Voice Activity Detection (silero-vad)
- **Watch mode** — always-on background listening for voice command triggers
- **GPU-accelerated** (CUDA) with automatic CPU fallback
- **Audio feedback** — start/stop sounds via PipeWire/PulseAudio
- **Systemd service** — runs in background, auto-restarts on crash
- **Fully offline** — no internet required after initial model download

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [silero-vad](https://github.com/snakers4/silero-vad).

## Configuration

All settings are in `~/.config/okawhisp/config.toml` (created by the installer).

```toml
[recording]
key = "ALT_GR"
model = "large-v3"
language = "de"
engine = "faster"

[vad]
enabled = true
threshold = 0.5
min_silence_ms = 2500

[watch]
idle_auto_close_seconds = 60
max_segment_ms = 10000
silence_ms = 1200

[actions]
action_cooldown_ms = 3000

[[actions.triggers]]
name = "type_here"
match = ["type here", "write here"]
command = "/bin/bash"
args = ["-lc", "xdotool type --delay 5 --clearmodifiers \"$OKAWISP_TEXT\""]
```

See `config.example.toml` for all options. Changes take effect on `systemctl --user restart okawhisp`.

## Models

| Model | Size | VRAM | Quality |
|-------|------|------|---------|
| tiny | 75 MB | 1 GB | Basic |
| base | 145 MB | 2 GB | Good |
| small | 470 MB | 4 GB | High |
| medium | 1.5 GB | 6 GB | Very high |
| large-v3 | 3 GB | 8 GB | Best |

The installer selects the best model for your GPU automatically. CPU-only systems use `tiny` or `base` with int8 quantization.

## Usage

```bash
# View live logs
journalctl --user -u okawhisp -f

# Restart after config changes
systemctl --user restart okawhisp

# Service status
systemctl --user status okawhisp
```

### CLI

Control the running service with `okawhisp`:

```bash
okawhisp status                 # daemon status
okawhisp watch start            # enable watch mode
okawhisp watch stop             # disable watch mode
okawhisp watch duration 5m      # extend watch for 5 minutes
okawhisp model status           # show active engine/model
okawhisp model set faster large-v3  # switch model and restart
okawhisp test trigger type_here --text "hello"  # test a trigger
```

## Troubleshooting

| Problem | Check |
|---------|-------|
| No audio input | `pactl list sources short` — verify mic is listed and not muted |
| Text not typed | `xdotool getactivewindow` — requires X11, not Wayland |
| No GPU | `python3 -c "import torch; print(torch.cuda.is_available())"` |
| No sounds | Ensure `paplay` is available (`pulseaudio-utils`) |

## License

MIT
