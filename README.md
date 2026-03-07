# okawhisp

**Fast local speech-to-text (STT) for Linux.** Press a hotkey to start speaking. Recording stops automatically when silence is detected, and your words are typed into any focused window — terminal, editor, chat, browser, anything.

You can also use a hold-to-talk flow: hold the key while speaking and send the transcription automatically on release/silence (depending on your configuration).

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, 4x faster than OpenAI Whisper) and [silero-vad](https://github.com/snakers4/silero-vad) for ML-based silence detection. Fully local, no cloud, runs on your GPU or CPU.

---

## Features

- **Global hotkey** (**AltGr** by default) — works in any window
- **Configurable hotkey** — supports AltGr plus F1–F12 via config/CLI
- **One-key flow** — press the hotkey to start recording; speech end/silence triggers automatic stop + send
- **Auto-stop** via Voice Activity Detection (silero-vad) — stops when you stop talking
- **Audio ducking** — background audio fades during recording and is restored afterward
- **GPU-accelerated** transcription (CUDA) with CPU fallback
- **Systemd service** — runs in the background, auto-restarts on crash

---

## Installation

```bash
curl -sSL https://github.com/okahari/okawhisp/raw/main/install.sh | bash
```

---

## Configuration

Create `~/.config/okawhisp/config.toml`:

```toml
key = "ALT_GR"   # Alternatives: F1..F12
model = "large-v3"
language = "de"
engine = "faster"
beam_size = 5
silence = 2.5
prompt = "NestJS, Flutter, Kubernetes"
```

Or set environment variables:

```bash
VAD_ENABLED=True                  # Enable silero-vad (default: True)
VAD_THRESHOLD=0.5                 # Speech probability (0.0–1.0)
VAD_MIN_SILENCE_MS=2500           # Silence before stop (ms)
DUCK_AUDIO_DURING_RECORDING=True  # Fade background audio
```

---

## Model Comparison

| Model | Download | VRAM | Quality | Speed (GPU) |
|-------|----------|------|---------|-------------|
| tiny | 75 MB | 1 GB | Good | Very Fast |
| base | 145 MB | 2 GB | Better | Fast |
| small | 470 MB | 4 GB | High | Medium |
| medium | 1.5 GB | 6 GB | Very High | Slower |
| large-v3 | 3 GB | 8 GB | Best | Slower |

CPU inference is supported (int8 quantization). Use `tiny` or `base` for CPU-only systems.

---

## Usage

Service runs automatically after install. Press **AltGr** to record.

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
