# voice-type 🎤

**System-level voice-to-text for Linux.** Press F9, speak, and your words appear in any focused window — terminal, editor, chat, browser, anything.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription and [silero-vad](https://github.com/snakers4/silero-vad) for smart silence detection. No cloud, no subscription — runs fully local on your GPU.

---

## Features

- **Global hotkey** (F9 by default) — works in any window
- **Auto-stop** via ML-based Voice Activity Detection (silero-vad) — stops when you stop talking, not on a fixed timer
- **Audio ducking** — background music fades to 10% during recording, restores afterward
- **GPU-accelerated** transcription via faster-whisper (CUDA) or CPU fallback
- **Bluetooth mic support** — handles Jabra and other BT headsets with persistent PyAudio stream
- **Clipboard fallback** — text goes to clipboard if no window is focused (via gpaste-client, xclip, xsel, or wl-copy)
- **Custom sounds** — optional MP3 start/stop sounds
- **Systemd service** — runs in the background, auto-restarts on crash

---

## Requirements

### Hardware
- Microphone (USB, 3.5mm, or Bluetooth)
- NVIDIA GPU recommended (CUDA) — CPU works but is slower

### System
- Linux with X11 (Wayland support via XWayland)
- PipeWire or PulseAudio
- `xdotool` for text injection
- `pactl` for audio ducking
- One of: `gpaste-client`, `xclip`, `xsel`, or `wl-copy` for clipboard

### Python
- Python 3.10+
- See `requirements.txt`

---

## Installation

### 1. Install system dependencies

```bash
sudo apt install xdotool xclip python3-dev portaudio19-dev
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA is recommended for fastest transcription.
> Install it separately if needed: https://pytorch.org/get-started/locally/

### 4. Run

```bash
python voice-type.py
```

Press **F9** to start recording. Speak. It stops automatically when you stop talking.

---

## Usage

```
python voice-type.py [OPTIONS]

Options:
  --key F9            Hotkey (F1–F12, default: F9)
  --model large-v3    Whisper model: tiny/base/small/medium/large-v3 (default: medium)
  --language de       Language code or "auto" for auto-detect (default: de)
  --engine faster     Engine: faster (recommended) or openai (default: faster)
  --beam-size 5       Beam search size: 1=fast, 5=accurate (default: 5)
  --silence 2.0       Fallback silence duration in seconds (default: 2.0)
  --prompt "..."      Context hint for better recognition of technical terms
```

### Examples

```bash
# English, fast model
python voice-type.py --language en --model small

# German, best quality
python voice-type.py --language de --model large-v3 --beam-size 5

# With domain hints for better accuracy
python voice-type.py --prompt "Python, FastAPI, Kubernetes, TypeScript"
```

---

## Configuration

Edit the config section at the top of `voice-type.py`:

```python
# silero-vad (Voice Activity Detection)
VAD_ENABLED = True          # True = silero-vad, False = RMS fallback
VAD_THRESHOLD = 0.5         # Speech probability threshold (0.0–1.0)
VAD_MIN_SILENCE_MS = 2500   # Silence after speech before stopping (ms)

# Audio ducking
DUCK_AUDIO_DURING_RECORDING = True
DUCK_SINK_LEVEL = 10        # Reduce background audio to this % during recording

# Custom sounds (optional)
CUSTOM_RECORD_START_SOUND = None  # e.g., "/home/user/sounds/start.mp3"
CUSTOM_RECORD_END_SOUND = None    # e.g., "/home/user/sounds/stop.mp3"

# Recording limits
MIN_RECORD_SECONDS = 1.0
MAX_RECORD_SECONDS = 120
```

---

## Systemd Service (autostart)

Copy the example service file and adjust the paths:

```bash
cp voice-type.service.example ~/.config/systemd/user/voice-type.service
```

Edit the file and replace:
- `/path/to/venv/bin/python` → your Python venv path
- `/path/to/voice-type.py` → absolute path to the script
- `YOUR_USER` → your Linux username
- `YOUR_UID` → your user ID (`id -u`)

Then enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now voice-type.service
systemctl --user status voice-type.service
```

Logs:

```bash
journalctl --user -u voice-type.service -f
# or
tail -f ~/.local/share/voice-type/voice-type.log
```

---

## How It Works

1. **F9 pressed** → stream starts, background audio fades to 10%
2. **Start sound** plays as the GO signal
3. **Recording** — silero-vad analyzes 512-sample chunks (32ms each) in real time
4. **Auto-stop** — after `VAD_MIN_SILENCE_MS` ms of silence following speech
5. **Transcription** — faster-whisper converts audio to text (GPU-accelerated)
6. **Output** — text is typed into the focused window via xdotool; also copied to clipboard
7. **Unduck** — background audio fades back to original volume, stop sound plays

---

## Troubleshooting

### No audio input
```bash
pactl list sources short  # check available input devices
```

### Bluetooth mic not working
The script keeps a persistent PyAudio stream to avoid Bluetooth SCO reconnect delays. If the mic stops working, restart the service:
```bash
systemctl --user restart voice-type.service
```

### Text not appearing
Make sure `xdotool` is installed and a window is focused. Check:
```bash
xdotool getactivewindow
```

### GPU not used
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Install CUDA-enabled PyTorch if needed.

---

## License

MIT — see [LICENSE](LICENSE)
