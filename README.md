# okawhisp

**Fast local speech-to-text (STT) for Linux.** Press F9, speak, and your words appear in any focused window — terminal, editor, chat, browser, anything.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, 4x faster than OpenAI Whisper) and [silero-vad](https://github.com/snakers4/silero-vad) for ML-based silence detection. Fully local, no cloud, runs on your GPU or CPU.

---

## Features

- **Global hotkey** (F9 by default) — works in any window
- **Auto-stop** via Voice Activity Detection (silero-vad) — stops when you stop talking
- **Audio ducking** — background music fades during recording, restores afterward
- **GPU-accelerated** transcription (CUDA) with CPU fallback
- **Bluetooth mic support** — persistent PyAudio stream for reliable BT headsets
- **Clipboard fallback** — uses gpaste-client, xclip, xsel, or wl-copy
- **Systemd service** — runs in the background, auto-restarts on crash

---

## Installation

```bash
curl -sSL https://github.com/username/okawhisp/raw/main/install.sh | bash
```

That's it. The installer handles everything: dependencies, systemd service, GPU detection.

---

## Configuration

Create `~/.config/okawhisp/config.toml`:

```toml
key = "F9"
model = "large-v3"
language = "de"
engine = "faster"
beam_size = 5
silence = 2.5
prompt = "NestJS, Flutter, Kubernetes"
```

Or set environment variables:

```bash
VAD_ENABLED=True              # Enable silero-vad (default: True)
VAD_THRESHOLD=0.5             # Speech probability (0.0–1.0)
VAD_MIN_SILENCE_MS=2500       # Silence before stop (ms)
DUCK_AUDIO_DURING_RECORDING=True  # Fade background audio
```

---

## Model Comparison

| Model | Size | VRAM | Quality | Speed (GPU) |
|-------|------|------|---------|-------------|
| tiny | 39 MB | 1 GB | Good | Very Fast |
| base | 74 MB | 2 GB | Better | Fast |
| small | 244 MB | 4 GB | High | Medium |
| medium | 769 MB | 6 GB | Very High | Slower |
| large-v3 | 1.5 GB | 8 GB | Best | Slower |

CPU inference supported (int8 quantization). Use `tiny` or `base` for CPU-only systems.

---

## Usage

Service runs automatically after install. Press **F9** to record.

Logs:
```bash
journalctl --user -u okawhisp.service -f
# or
tail -f ~/.local/share/okawhisp/okawhisp.log
```

Restart service:
```bash
systemctl --user restart okawhisp.service
```

---

## How It Works

1. **F9 pressed** → stream starts, background audio ducking enabled
2. **Start sound** plays
3. **Recording** — silero-vad processes 512-sample chunks (32ms each)
4. **Auto-stop** — after configured silence duration following speech
5. **Transcription** — faster-whisper converts audio to text
6. **Output** — xdotool types text into focused window + clipboard fallback
7. **Unduck** — background audio restored, stop sound plays

---

## Troubleshooting

**No audio input:**
```bash
pactl list sources short
```

**Bluetooth mic issues:**
```bash
systemctl --user restart okawhisp.service
```

**Text not appearing:**
Check xdotool: `xdotool getactivewindow`

**GPU not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## License

MIT — see [LICENSE](LICENSE)
