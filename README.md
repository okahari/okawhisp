# OkaWhisp 🎙️

**Local voice typing for Linux** — speak and your words appear anywhere.

> Speech-to-text powered by Whisper. Push-to-talk or toggle mode. Fully offline, privacy-first.

Press a hotkey → speak → text is typed into any focused window (terminal, browser, chat, editor — anything).

- **Push-to-Talk (PTT)**: Hold the key while speaking, release to transcribe & send
- **Toggle mode**: Tap to start recording, silence auto-stops
- **Watch mode**: Always-on background listening for trigger words (voice commands)

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (4× faster than OpenAI Whisper) and [silero-vad](https://github.com/snakers4/silero-vad) for ML-based voice activity detection. Runs **100% locally** on your GPU or CPU — no cloud, no API keys, no data leaves your machine.

---

## Features

- 🎹 **Global hotkey** (AltGr default, F1–F12 configurable) — works in any window
- 🎤 **Push-to-Talk + Toggle mode** — hold to talk or tap to start/stop
- 🤫 **Smart auto-stop** via Voice Activity Detection (silero-vad)
- 👂 **Watch mode** — always-on trigger word detection for voice commands
- ⚡ **GPU-accelerated** (CUDA) with CPU fallback
- 🔊 **Audio feedback** — start/stop sounds routed through PipeWire/PulseAudio
- 🔄 **Systemd service** — runs in background, auto-restarts on crash
- 🔒 **Offline & private** — no internet required after model download

---

## Dependencies

### System packages

```bash
# Debian/Ubuntu
sudo apt install xdotool portaudio19-dev python3-dev

# Fedora
sudo dnf install xdotool portaudio-devel python3-devel

# Arch
sudo pacman -S xdotool portaudio python
```

Optional: `paplay` (from `pulseaudio-utils` or PipeWire) for sound feedback.

### Python packages

```
torch numpy pyaudio faster-whisper silero-vad pynput
```

Install via:
```bash
pip install --user torch numpy pyaudio faster-whisper silero-vad pynput
```

---

## Installation

```bash
curl -sSL https://github.com/okahari/okawhisp/raw/main/install.sh | bash
```

The installer handles system packages, Python dependencies, model download, and systemd service setup automatically.

---

## Configuration

Create `~/.config/okawhisp/config.toml`:

```toml
[recording]
key = "ALT_GR"      # Alternatives: F1..F12
model = "large-v3"
language = "de"
engine = "faster"

[watch]
idle_auto_close_seconds = 60   # 0 = disable watch mode
max_segment_ms = 10000
silence_ms = 1200
min_segment_ms = 600

[sounds]
# start = "/path/to/start.wav"   # custom start sound (optional)
# stop  = "/path/to/stop.wav"    # custom stop sound (optional)

[actions]
action_cooldown_ms = 3000

[[actions.triggers]]
name = "type_here"
match = ["type here", "write here"]
command = "/bin/bash"
args = ["-lc", "xdotool type --delay 5 --clearmodifiers \"$OKAWISP_TEXT\""]
```

See `config.example.toml` for all available options.

---

## Models

| Model | Size | VRAM | Quality | Speed |
|-------|------|------|---------|-------|
| tiny | 75 MB | 1 GB | Good | ⚡⚡⚡ |
| base | 145 MB | 2 GB | Better | ⚡⚡ |
| small | 470 MB | 4 GB | High | ⚡ |
| medium | 1.5 GB | 6 GB | Very High | 🐢 |
| large-v3 | 3 GB | 8 GB | Best | 🐢 |

CPU inference supported (int8). Use `tiny` or `base` for CPU-only systems.

---

## Usage

Service starts automatically after install. Press **AltGr** (or your configured key) to record.

```bash
# View logs
journalctl --user -u okawhisp.service -f

# Or check the log file directly
tail -f ~/.local/share/okawhisp/okawhisp.log

# Restart
systemctl --user restart okawhisp.service
```

### Watch mode

When enabled (default), okawhisp listens in the background for trigger words defined in `[[actions.triggers]]`. The microphone closes automatically after the idle timeout to preserve privacy.

Use `okawhispctl` to control watch mode:
```bash
okawhispctl watch stop          # disable watch mode
okawhispctl watch duration 5m   # extend watch for 5 minutes
```

---

## Troubleshooting

| Problem | Check |
|---------|-------|
| No audio input | `pactl list sources short` — verify your mic is listed |
| Text not typed | `xdotool getactivewindow` — must be on X11 (not Wayland) |
| No GPU | `python3 -c "import torch; print(torch.cuda.is_available())"` |
| No start/stop sounds | Ensure `paplay` is installed (`pulseaudio-utils`) |

---

## License

MIT — see [LICENSE](LICENSE)

---

<sub>**Keywords:** voice typing linux, speech to text, whisper, faster-whisper, STT, voice input, dictation, push to talk, PTT, voice recognition, transcription, offline speech recognition, local AI, silero-vad, voice activity detection</sub>
