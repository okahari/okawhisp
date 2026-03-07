# OkaWhisp 🎙️

**Local voice typing for Linux** — speak and your words appear anywhere.

> Speech-to-text powered by Whisper. Push-to-talk or toggle mode. Fully offline, privacy-first.

Press a hotkey → speak → text is typed into any focused window (terminal, browser, chat, editor — anything).

- **Push-to-Talk (PTT)**: Hold the key while speaking, release to transcribe & send
- **Toggle mode**: Tap to start recording, silence auto-stops

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (4× faster than OpenAI Whisper) and [silero-vad](https://github.com/snakers4/silero-vad) for ML-based voice activity detection. Runs **100% locally** on your GPU or CPU — no cloud, no API keys, no data leaves your machine.

---

## Features

- 🎹 **Global hotkey** (AltGr default, F1–F12 configurable) — works in any window
- 🎤 **Push-to-Talk + Toggle mode** — hold to talk or tap to start/stop
- 🤫 **Smart auto-stop** via Voice Activity Detection (silero-vad)
- 🔇 **Audio ducking** — background music fades during recording
- ⚡ **GPU-accelerated** (CUDA) with CPU fallback
- 🔄 **Systemd service** — runs in background, auto-restarts on crash
- 🔒 **Offline & private** — no internet required after model download

---

## Installation

```bash
curl -sSL https://github.com/okahari/okawhisp/raw/main/install.sh | bash
```

---

## Configuration

Create `~/.config/okawhisp/config.toml`:

```toml
[recording]
key = "ALT_GR"      # Alternatives: F1..F12
model = "large-v3"
language = "de"
engine = "faster"
beam_size = 5
silence = 2.5
prompt = ""         # Optional: "PostgreSQL, Kubernetes, Tailwind"

[vad]
enabled = true
threshold = 0.5
min_silence_ms = 2500

[duck]
enabled = true
```

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

Service starts automatically. Press **AltGr** (or your configured key) to record.

```bash
# View logs
journalctl --user -u okawhisp.service -f

# Restart
systemctl --user restart okawhisp.service
```

---

## Troubleshooting

| Problem | Check |
|---------|-------|
| No audio | `pactl list sources short` |
| Text not typed | `xdotool getactivewindow` |
| No GPU | `python -c "import torch; print(torch.cuda.is_available())"` |

---

## License

MIT — see [LICENSE](LICENSE)

---

<sub>**Keywords:** voice typing linux, speech to text, whisper, faster-whisper, STT, voice input, dictation, push to talk, PTT, voice recognition, transcription, offline speech recognition, local AI, silero-vad, voice activity detection</sub>
