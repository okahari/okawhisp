#!/usr/bin/env python3
# Requirements: numpy, faster-whisper, silero-vad>=6.0, torch, pynput
# System: parec (pulseaudio-utils) for audio capture
# Install via: python3 -m pip install --user numpy faster-whisper silero-vad torch pynput
"""
OkaWhisp - System-Level Voice-to-Text for Any Window
=====================================================

Global hotkey → record microphone → silence detection auto-stops
→ Whisper transcribes → text is typed directly into the active window

Works with Claude Code, any terminal, any editor.

Engines:
  - faster-whisper (default): 4x faster than OpenAI Whisper, CTranslate2
  - openai-whisper: Original OpenAI implementation (local)
  - api: OpenAI-compatible API (OpenAI, Groq, local Whisper server, etc.)

Config File (optional):
  ~/.config/okawhisp/config.toml  →  set defaults, no CLI needed

Usage:
  python okawhisp.py                          # Default: AltGr, medium, German
  python okawhisp.py --key F8                 # Different hotkey
  python okawhisp.py --key ALT_GR             # Explicit AltGr hotkey
  python okawhisp.py --model small            # Smaller model
  python okawhisp.py --language en            # English
  python okawhisp.py --engine openai          # Local OpenAI Whisper
  python okawhisp.py --engine api             # OpenAI-compatible API
  python okawhisp.py --api-url https://api.groq.com/openai/v1 --api-key KEY
  python okawhisp.py --prompt "NestJS, Flutter"  # Context hints
"""

import subprocess
import threading
import time
import sys
import signal
import argparse
import warnings
import os

import shutil

if not shutil.which("parec"):
    print("❌ parec not found. Run: sudo apt install pulseaudio-utils")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("❌ NumPy not installed. Run: pip install --user numpy")
    sys.exit(1)
import io
import wave
import logging
import json
import socket
import re
import queue as _queue_mod
import select
from pathlib import Path
from datetime import datetime
try:
    import tomllib          # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib   # pip install tomli (fallback for Python < 3.11)
    except ImportError:
        tomllib = None
from ctypes import cdll, c_char_p, c_int, CFUNCTYPE

warnings.filterwarnings("ignore")

# ─── Logging Setup ───────────────────────────────────────────────

LOG_FILE = os.path.expanduser("~/.local/share/okawhisp/okawhisp.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

_log_formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("okawhisp")
log.setLevel(logging.DEBUG)

# File handler (rotating: max 2MB, 3 backups)
from logging.handlers import RotatingFileHandler
_fh = RotatingFileHandler(LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(_log_formatter)
_fh.setLevel(logging.DEBUG)
log.addHandler(_fh)

# Console handler (INFO+ only)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_log_formatter)
_ch.setLevel(logging.INFO)
log.addHandler(_ch)

# ─── Configuration ───────────────────────────────────────────────

# Whisper requirements (final target values)
SAMPLE_RATE = 16000         # Whisper requires 16kHz
CHUNK_SIZE = 1024           # ~64ms per chunk (at 16kHz)

# Input device override (read from config.toml [recording] section).
# None = PulseAudio/PipeWire default source. Can be a PulseAudio source name
# (e.g. "bluez_input.XX:XX:XX:XX:XX:XX" for a specific Bluetooth mic).
INPUT_DEVICE = None

MODEL_SIZE = "medium"       # medium = best balance for RTX 3060 Ti
LANGUAGE = "de"             # German as default (None = auto-detect)
ENGINE = "faster"           # faster-whisper or openai
BEAM_SIZE = 5               # Beam search for better quality
INITIAL_PROMPT = None       # Context prompt for technical terms
SILENCE_THRESHOLD = 200     # RMS threshold for silence (Jabra BT: speech ~220 RMS)
SILENCE_DURATION = 2.0      # Seconds of silence until auto-stop (RMS fallback)
MIN_RECORD_SECONDS = 1.0    # Minimum recording duration
MAX_RECORD_SECONDS = 120    # No-speech timeout (recording stops if no speech within this)
MAX_RECORD_SECONDS_ABSOLUTE = 360  # Hard limit — even with active speech (6 min)

# ── silero-vad (Voice Activity Detection) ───────────────────────
# Replaces RMS threshold with ML-based speech recognition.
# No calibration tuning needed — works automatically.
VAD_ENABLED = True          # True = silero-vad, False = RMS fallback
VAD_THRESHOLD = 0.5         # Speech probability threshold (0.0–1.0)
VAD_MIN_SILENCE_MS = 2500   # Silence after last speech until auto-stop
VAD_MIN_SPEECH_MS  = 200    # Minimum speech before stop counter activates
VAD_CHUNK_SAMPLES  = 512    # silero requires exactly 512 samples at 16kHz (32ms)


# Custom sounds (played if available)
# Sounds: automatically loaded from ~/.local/share/okawhisp/sounds/ (by installer)
_SOUND_DIR = Path.home() / ".local" / "share" / "okawhisp" / "sounds"
CUSTOM_RECORD_START_SOUND = str(_SOUND_DIR / "start.mp3") if (_SOUND_DIR / "start.mp3").exists() else None
CUSTOM_RECORD_END_SOUND = str(_SOUND_DIR / "stop.mp3") if (_SOUND_DIR / "stop.mp3").exists() else None

# ── OpenAI-compatible API (engine=api) ───────────────────────────
# Works with: OpenAI, Groq, local whisper.cpp server, etc.
WHISPER_API_BASE_URL = "https://api.openai.com/v1"
WHISPER_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
WHISPER_API_MODEL    = "whisper-1"

# ─── Config File ─────────────────────────────────────────────────

def load_config() -> dict:
    """Load ~/.config/okawhisp/config.toml (or ~/.okawhisp.toml).

    Returns empty dict if no config file found or TOML not available.
    CLI arguments always take precedence over config file values.

    Example config.toml:
        [recording]
        key = "ALT_GR"
        model = "large-v3"
        language = "de"
        engine = "faster"

        [vad]
        enabled = true
        threshold = 0.5
        min_silence_ms = 2500

        [sounds]
        start = "/pfad/zu/start.mp3"
        stop  = "/pfad/zu/stop.mp3"

        [api]
        base_url = "https://api.groq.com/openai/v1"
        api_key  = "gsk_..."
        model    = "whisper-large-v3"
    """
    if tomllib is None:
        return {}
    config_paths = [
        Path.home() / ".config" / "okawhisp" / "config.toml",
        Path.home() / ".okawhisp.toml",
    ]
    for path in config_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    cfg = tomllib.load(f)
                log.info(f"  ⚙️  Config loaded: {path}")
                return cfg
            except Exception as e:
                log.warning(f"  ⚠️  Config error ({path}): {e}")
    return {}

# ─── Global Variables ────────────────────────────────────────────

model = None
should_exit = False
engine_type = "faster"
_vad_model = None           # silero-vad model (loaded at startup)

# Lock instead of bool: prevents race condition on rapid hotkey presses
# (bool check + set is not atomic).
_recording_lock = threading.Lock()
_ptt_stop_requested = False  # PTT mode: stop when key released

# Persistent parec subprocess; opened/closed on demand.
# Starts "warm" and closes after idle timeout (privacy + GNOME mic indicator).
_audio_proc = None       # subprocess.Popen running parec
_audio_stream = None     # alias for _audio_proc (record functions use this)

# Idle close policy (requested): 60s after app start and 60s after each recording stop.
IDLE_CLOSE_SECONDS = 60
_idle_close_timer = None
_idle_timer_lock = threading.Lock()

CONTROL_SOCKET_PATH = str(Path.home() / ".local" / "share" / "okawhisp" / "control.sock")

WATCH_ACTIVE = True
WATCH_UNTIL_TS = None

ACTION_TRIGGERS = []
ACTION_COOLDOWN_MS = 2500
AUTO_PROMPT_FROM_TRIGGERS = True

# Trigger matching guardrails
CONTEXT_MATCH_WATCH_ONLY = True
REQUIRE_COMMAND_PREFIX = True

_trigger_last_fire = {}

WATCH_MAX_SEGMENT_MS = 10000
WATCH_SILENCE_MS     = 1200  # VAD silence after last speech to end a watch segment
WATCH_MIN_SEGMENT_MS = 600   # Minimum segment length to bother transcribing
WATCH_SUSPENDED = False

# Queue for non-blocking watch-mode transcription
_watch_transcribe_queue = _queue_mod.Queue(maxsize=4)

# ─── Trigger payload-extraction patterns (shared across watch + record paths) ──
# Removing the command phrase before typing the remaining text.
_PAT_TYPE_TO_ACTIVE = (
    r"\b(?:type|write|tippe?|schreib(?:e)?|diktier(?:e)?)\b"
    r"\s*(?:(?:ins?|das)\s+)?(?:aktiv\w*\s+)?(?:hier|here|direkt|aktiv\w*|fenster|window)\b"
)
_PAT_TYPE_TO_TELEGRAM = (
    r"\b(?:send|type|write|sende|schick(?:e|en)?|übertrag(?:e|en)?|übermittle|schreib(?:e)?)\b"
    r"\s*(?:an\s+|zu\s+|nach\s+|to\s+)?\btelegramm?\b"
    r"|\b(?:go\s+to|open)\b\s*\btelegramm?\b"
)

# Signals for event-based chime timing
_recording_ready_event = threading.Event()
_recording_stopped_event = threading.Event()

# ─── ALSA error suppression (once at startup) ──────────────
# IMPORTANT: Set only ONCE, not on every recording!
# Setting multiple times causes SEGV crashes
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except (OSError, AttributeError):
    pass  # ALSA not available (Wayland-only, non-Linux, etc.)


def load_model_faster_whisper(model_size, device):
    """Load faster-whisper model (4x faster than OpenAI)"""
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    m = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
    )
    return m


def load_model_openai_whisper(model_size, device):
    """Load original OpenAI Whisper model"""
    import whisper
    return whisper.load_model(model_size, device=device)


def load_vad_model():
    """Loads the silero-vad model (runs on CPU — tiny model, ~1MB).

    Returns the model or None if import fails.
    """
    global _vad_model
    try:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
        _vad_model.eval()
        log.info("  ✅ silero-vad loaded (VAD active)")
        return _vad_model
    except Exception as e:
        log.warning(f"  ⚠️  silero-vad not available: {e} → RMS-Fallback")
        return None


WHISPER_SERVER_URL = "http://127.0.0.1:8181"


def transcribe_via_server(audio_np):
    """Transcription via whisper-server (GPU, shared). Returns None if not reachable."""
    try:
        import urllib.request
        import base64
        import json

        audio_b64 = base64.b64encode(audio_np.tobytes()).decode()
        payload = json.dumps({
            "audio_b64": audio_b64,
            "language": LANGUAGE or "auto",
            "beam_size": BEAM_SIZE,
        }).encode()

        req = urllib.request.Request(
            f"{WHISPER_SERVER_URL}/transcribe_numpy",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get("text", "").strip(), result.get("language", "?")
    except Exception:
        return None


def transcribe_faster_whisper(audio_np):
    """Transcription with faster-whisper — prefers whisper-server (GPU shared),
    falls back to local instance if server not reachable."""

    # Try server first (model already loaded, GPU)
    result = transcribe_via_server(audio_np)
    if result is not None:
        return result

    # Local fallback (model must be loaded)
    if model is None:
        return "", "?"

    segments, info = model.transcribe(
        audio_np,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=300,
        ),
        no_speech_threshold=0.5,
        condition_on_previous_text=True,
    )

    text = " ".join(segment.text.strip() for segment in segments)
    detected_lang = info.language if info else "?"
    return text.strip(), detected_lang


def transcribe_openai_whisper(audio_np):
    """Transcription with original OpenAI Whisper"""
    result = model.transcribe(
        audio_np,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        initial_prompt=INITIAL_PROMPT,
        fp16=True if hasattr(model, 'device') and model.device.type == "cuda" else False,
        no_speech_threshold=0.5,
        condition_on_previous_text=True,
    )
    return result['text'].strip(), result.get('language', '?')


def transcribe_via_api(audio_np):
    """Transcription via OpenAI-compatible API.

    Works with:
      - OpenAI (api.openai.com)        → WHISPER_API_MODEL = "whisper-1"
      - Groq   (api.groq.com/openai/v1) → WHISPER_API_MODEL = "whisper-large-v3"
      - local whisper.cpp server        → WHISPER_API_BASE_URL = "http://localhost:8080"
      - any other OpenAI-compatible endpoint

    Configuration via config file [api] or --api-url / --api-key CLI args.
    API key can also be set as environment variable OPENAI_API_KEY.
    """
    try:
        import urllib.request
        import urllib.error
        import json

        # Write audio as WAV to in-memory buffer
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())
        wav_buf.seek(0)
        audio_bytes = wav_buf.read()

        # Build multipart/form-data manually (no requests needed)
        boundary = "voicetype_boundary_xk29"
        def field(name, value):
            return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                    f'{value}\r\n').encode()
        def file_field(name, filename, content_type, data):
            header = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                      f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n').encode()
            return header + data + b'\r\n'

        body = (field("model", WHISPER_API_MODEL) +
                field("response_format", "json"))
        if LANGUAGE:
            body += field("language", LANGUAGE)
        if INITIAL_PROMPT:
            body += field("prompt", INITIAL_PROMPT)
        body += file_field("file", "audio.wav", "audio/wav", audio_bytes)
        body += f"--{boundary}--\r\n".encode()

        url = f"{WHISPER_API_BASE_URL.rstrip('/')}/audio/transcriptions"
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if WHISPER_API_KEY:
            headers["Authorization"] = f"Bearer {WHISPER_API_KEY}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            text = result.get("text", "").strip()
            lang = result.get("language", LANGUAGE or "?")
            return text, lang

    except Exception as e:
        log.error(f"  [api] ❌ API transcription failed: {e}")
        raise


def _play_pcm_sound(samples, sample_rate=44100):
    """Plays a short PCM signal via paplay (routes through PipeWire correctly)."""
    if samples is None or len(samples) == 0:
        return False

    try:
        pcm = np.clip(samples, -1.0, 1.0)
        pcm_bytes = (pcm * 32767).astype(np.int16).tobytes()

        # Write WAV to temp file and play via paplay
        wav_path = os.path.join(os.path.dirname(LOG_FILE), ".tmp_sound.wav")
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        subprocess.Popen(
            ['paplay', wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        log.warning(f"  Sound: PCM playback failed: {e}")
        return False


def _play_audio_file(file_path, blocking: bool = False):
    """Play an audio file via paplay (PulseAudio/PipeWire).

    paplay routes through PipeWire → correct output device regardless of
    which sink is active (headset, speakers, HDMI).
    """
    if not file_path or not os.path.isfile(file_path):
        return False

    # paplay handles WAV/OGA natively; convert MP3 to WAV if needed.
    play_path = file_path
    if file_path.lower().endswith('.mp3'):
        wav_path = file_path.rsplit('.', 1)[0] + '.wav'
        if not os.path.isfile(wav_path):
            try:
                subprocess.run(
                    ['ffmpeg', '-i', file_path, '-y', wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
            except Exception as e:
                log.warning(f"  Sound: ffmpeg convert failed: {e}")
                return False
        play_path = wav_path

    try:
        cmd = ['paplay', play_path]
        if blocking:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        log.warning("  Sound: paplay not found")
        return False


def _switch_click_sound(sample_rate=44100):
    """Short mechanical switch click for recording start."""
    duration = 0.06
    length = int(sample_rate * duration)
    signal = np.zeros(length, dtype=np.float32)

    def add_pulse(start_ms, freq, amp, decay):
        start = int(sample_rate * (start_ms / 1000.0))
        pulse_len = int(sample_rate * 0.02)
        if start >= length:
            return
        end = min(length, start + pulse_len)
        t = np.arange(end - start) / sample_rate
        pulse = amp * np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)
        signal[start:end] += pulse.astype(np.float32)

    # Double pulse creates the "switch" character instead of a simple beep.
    add_pulse(start_ms=0.0, freq=2200, amp=0.55, decay=170)
    add_pulse(start_ms=9.0, freq=1350, amp=0.40, decay=180)

    # Slight noise trace for haptic click impression.
    noise_len = int(sample_rate * 0.012)
    if noise_len > 0:
        noise = np.random.normal(0, 1, noise_len).astype(np.float32)
        noise *= np.exp(-np.linspace(0, 12, noise_len)).astype(np.float32)
        signal[:noise_len] += noise * 0.12

    return signal * 0.38


def _soft_end_buzzer_sound(sample_rate=44100):
    """Soft electronic end tone (subtle buzzer)."""
    duration = 0.24
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate

    # Soft down-sweep: signals "end" without sounding harsh.
    freq = np.linspace(780.0, 620.0, n, dtype=np.float32)
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate

    main = np.sin(phase)
    detuned = np.sin(phase * 1.012)
    harmonic = np.sin(phase * 2.0)
    am = 0.9 + 0.1 * np.sin(2 * np.pi * 26.0 * t)

    signal = (0.68 * main + 0.22 * detuned + 0.10 * harmonic) * am

    attack = np.clip(t / 0.015, 0.0, 1.0)
    release = np.exp(-np.clip(t - 0.06, 0.0, None) * 11.5)
    envelope = attack * release

    return (signal * envelope * 0.30).astype(np.float32)


def _mic_error_sound(sample_rate=44100):
    """Short descending two-tone error signal: mic not ready / unavailable."""
    segments = []
    for freq in (880, 620):
        n = int(sample_rate * 0.07)
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t * 28.0)
        segments.append(np.sin(2 * np.pi * freq * t) * env * 0.38)
    gap = np.zeros(int(sample_rate * 0.015), dtype=np.float32)
    return np.concatenate([segments[0], gap, segments[1]])


def _startup_ready_sound(sample_rate=44100):
    """Short ascending three-note arpeggio: system ready / watch active."""
    segments = []
    for freq in (523, 659, 784):   # C5 → E5 → G5
        n = int(sample_rate * 0.08)
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t * 18.0)
        segments.append(np.sin(2 * np.pi * freq * t) * env * 0.32)
        segments.append(np.zeros(int(sample_rate * 0.025), dtype=np.float32))
    return np.concatenate(segments)


def play_sound(sound_name, blocking: bool = False):
    """Play feedback sound via paplay (PipeWire-routed).

    Priority: custom file → synthetic sound → system sound.
    """
    custom_file_map = {
        "record_start": CUSTOM_RECORD_START_SOUND,
        "record_end": CUSTOM_RECORD_END_SOUND,
    }

    # 1) Custom sound files (user-provided).
    if _play_audio_file(custom_file_map.get(sound_name), blocking=blocking):
        return

    # 2) Synthetic sounds as fallback.
    synthetic_sounds = {
        "record_start": _switch_click_sound,
        "record_end": _soft_end_buzzer_sound,
        "mic_error": _mic_error_sound,
        "startup_ready": _startup_ready_sound,
    }

    builder = synthetic_sounds.get(sound_name)
    if builder is not None and _play_pcm_sound(builder()):
        return

    # 3) System sounds (last resort).
    try:
        sound_path = f'/usr/share/sounds/freedesktop/stereo/{sound_name}.oga'
        if blocking:
            subprocess.run(['paplay', sound_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            subprocess.Popen(['paplay', sound_path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


def notify(title, message, urgency="normal"):
    """Send desktop notification (optional)"""
    try:
        subprocess.Popen(
            ['notify-send', '-u', urgency, '-t', '2000', title, message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass


def type_text(text):
    """Type text into the active window via xdotool"""
    try:
        subprocess.run(
            ['xdotool', 'type', '--delay', '10', '--clearmodifiers', text],
            check=True,
            timeout=30
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  xdotool Error: {e}")
        return False


def get_rms(data):
    """Calculate Root Mean Square of an audio chunk"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    if len(audio_data) == 0:
        return 0
    return np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))



def calibrate_silence_threshold(audio_stream, duration_s: float = 0.8) -> int:
    """Measure current noise floor and derive a dynamic threshold.

    Reads `duration_s` seconds of audio (~12 chunks at 0.8s) and uses the
    20th percentile of RMS values — more robust than median, since early speech
    or short outliers (single loud chunks) do not skew the value.

    Minimum: 40 (for very quiet environments).
    Maximum: 250 (safety cap).
    """
    n_chunks = max(1, int(duration_s * SAMPLE_RATE / CHUNK_SIZE))
    rms_values = []
    for _ in range(n_chunks):
        try:
            data = _read_chunk(audio_stream, CHUNK_SIZE)
            rms_values.append(get_rms(data))
        except Exception:
            pass
    if not rms_values:
        return SILENCE_THRESHOLD
    # 20th percentile: takes the quietest 20% of chunks → true silence floor
    # Median would be too high due to early speech.
    # Factor 4 (instead of 8): keeps threshold well below normal speech level (~150–400 RMS)
    noise_floor = float(np.percentile(rms_values, 20))
    dynamic = int(noise_floor * 4)
    dynamic = max(40, min(dynamic, 250))
    log.info(f"  [2b] Calibration: noise_floor(p20)={noise_floor:.1f} → threshold={dynamic} "
             f"(min={min(rms_values):.1f}, max={max(rms_values):.1f}, n={len(rms_values)})")
    return dynamic


def record_with_silence_detection(audio_stream, threshold: int = None):
    """Record audio until silence is detected"""
    effective_threshold = threshold if threshold is not None else SILENCE_THRESHOLD
    frames = []
    silent_chunks = 0
    # Calculations based on Whisper sample rate (16kHz)
    chunks_for_silence = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    min_chunks = int(MIN_RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
    # Warmup: silence counter only starts after 1.5s.
    # Gives the device time to stabilize AND ensures the
    # silence timer never runs before user hears the start sound.
    warmup_chunks = int(1.5 * SAMPLE_RATE / CHUNK_SIZE)
    total_chunks = 0

    while total_chunks < max_chunks:
        try:
            # Read 16kHz chunks
            data = _read_chunk(audio_stream, CHUNK_SIZE)
            frames.append(data)
            total_chunks += 1

            rms = get_rms(data)

            # Only count silence after warmup — always record before
            if total_chunks > warmup_chunks:
                if rms < effective_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0  # Sound → counter reset

            # Auto-stop bei Silence (nach Mindest-Recordingdauer)
            if total_chunks > min_chunks and silent_chunks >= chunks_for_silence:
                break

        except Exception as e:
            print(f"  Audio error: {e}")
            break

    return frames


def record_with_vad(audio_stream, ptt_mode: bool = False) -> list:
    """Record audio with silero-vad Voice Activity Detection.

    Reads 512-sample chunks (32ms at 16kHz), analyzes each via silero-vad.
    Stops when VAD_MIN_SILENCE_MS of silence follows detected speech.
    Returns all recorded bytes (incl. pre-speech frames) for Whisper.

    ptt_mode: If True, stop when _ptt_stop_requested becomes True (key released).

    State machine:
        WAITING  → waiting for first speech (but accumulating audio)
        SPEAKING → speech detected, silence counter running
        STOPPED  → silence timeout reached (or PTT key released)
    """
    import torch
    global _ptt_stop_requested

    vad_chunk = VAD_CHUNK_SAMPLES                            # 512 Samples = 32ms
    ms_per_chunk = vad_chunk / SAMPLE_RATE * 1000           # = 32ms
    silence_chunks_needed = int(VAD_MIN_SILENCE_MS / ms_per_chunk)
    min_speech_chunks     = int(VAD_MIN_SPEECH_MS  / ms_per_chunk)
    min_total_chunks      = int(MIN_RECORD_SECONDS * 1000 / ms_per_chunk)
    max_total_chunks      = int(MAX_RECORD_SECONDS * 1000 / ms_per_chunk)
    absolute_max_chunks   = int(MAX_RECORD_SECONDS_ABSOLUTE * 1000 / ms_per_chunk)

    frames          = []
    speech_chunks   = 0   # consecutive speech chunks
    silence_streak  = 0   # consecutive silence chunks after speech start
    speech_started  = False
    total_chunks    = 0

    while total_chunks < max_total_chunks and total_chunks < absolute_max_chunks:
        # PTT mode: stop immediately when key released
        if ptt_mode and _ptt_stop_requested:
            elapsed_ms = total_chunks * ms_per_chunk
            log.info(f"  PTT: 🛑 Key released → stop (t={elapsed_ms:.0f}ms)")
            break
        try:
            data = _read_chunk(audio_stream, vad_chunk)
        except Exception as e:
            log.warning(f"  VAD read-Error: {e}")
            break

        frames.append(data)
        total_chunks += 1

        # Float32 tensor [512] for silero-vad
        audio_f32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if total_chunks % 47 == 1:  # log every ~1.5s
            rms = float(np.sqrt(np.mean(audio_f32 ** 2)) * 32768)
            log.debug(f"  VAD[{total_chunks}] RMS={rms:.1f}")
        with torch.no_grad():
            speech_prob = _vad_model(torch.from_numpy(audio_f32), SAMPLE_RATE).item()

        is_speech = speech_prob >= VAD_THRESHOLD

        if is_speech:
            speech_chunks += 1
            silence_streak = 0
            # Extend timeout while speech is active
            new_max = total_chunks + int(MAX_RECORD_SECONDS * 1000 / ms_per_chunk)
            if new_max > max_total_chunks:
                max_total_chunks = min(new_max, absolute_max_chunks)
            if not speech_started and speech_chunks >= min_speech_chunks:
                speech_started = True
                elapsed_ms = total_chunks * ms_per_chunk
                log.info(f"  VAD: 🗣  Speech detected (prob={speech_prob:.2f}, t={elapsed_ms:.0f}ms)")
        else:
            speech_chunks = 0
            if speech_started:
                silence_streak += 1
                if (total_chunks >= min_total_chunks and
                        silence_streak >= silence_chunks_needed):
                    elapsed_ms = total_chunks * ms_per_chunk
                    log.info(f"  VAD: 🤫 Silence {silence_streak * ms_per_chunk:.0f}ms → stop "
                             f"(t={elapsed_ms:.0f}ms)")
                    break

        # Log progress every ~3s
        if total_chunks % 94 == 1:
            elapsed_ms = total_chunks * ms_per_chunk
            rms = float(np.sqrt(np.mean(audio_f32 ** 2)) * 32768)
            log.info(f"  VAD[{elapsed_ms/1000:.0f}s] prob={speech_prob:.3f} rms={rms:.0f} "
                     f"speech_started={speech_started}")

    # If we ran the full duration without speech, return empty
    if not speech_started:
        elapsed_s = total_chunks * ms_per_chunk / 1000
        log.warning(f"  VAD: ⚠ No speech detected in {elapsed_s:.0f}s → discarding")
        return []

    return frames


def _mic_device_info() -> str:
    """Returns recording device info (for logging)."""
    if INPUT_DEVICE:
        return f"parec (device={INPUT_DEVICE})"
    return "parec (system default mic)"


def _check_mic_ready(max_wait_s: float = 5.0, n_stable_reads: int = 3) -> bool:
    """Checks mic readiness: N consecutive successful reads with non-zero audio.

    Bluetooth mics often connect but deliver only zero-frames for several
    seconds while the HFP audio transport is still being established.
    We require at least one read with RMS > 0 to confirm real data flow.

    Returns True if ready, False on timeout.
    """
    deadline = time.time() + max_wait_s
    consecutive_ok = 0
    seen_nonzero = False
    attempts = 0

    while time.time() < deadline:
        attempts += 1
        try:
            data = _read_chunk(_audio_stream, CHUNK_SIZE)
            if data and len(data) == CHUNK_SIZE * 2:
                consecutive_ok += 1
                samples = np.frombuffer(data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                if rms > 0:
                    seen_nonzero = True
            else:
                consecutive_ok = 0
            log.debug(f"  MIC-CHECK read {attempts}: OK ({consecutive_ok}/{n_stable_reads}) nonzero={seen_nonzero}")
        except Exception as ex:
            consecutive_ok = 0
            log.warning(f"  MIC-CHECK read {attempts}: FAILED ({ex})")

        if consecutive_ok >= n_stable_reads and seen_nonzero:
            log.info(f"  ✅ Microphone ready after {attempts} read(s) ({n_stable_reads} consecutive OK, non-zero audio)")
            return True

    log.warning(f"  ⚠️  Microphone NOT ready after {max_wait_s:.1f}s ({attempts} attempts, nonzero={seen_nonzero})")
    _diagnose_mic_failure()
    return False


def _diagnose_mic_failure():
    """Log PipeWire/Bluetooth diagnostics when mic fails readiness check."""
    try:
        # Default source
        r = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True, timeout=3)
        default_src = r.stdout.strip() if r.returncode == 0 else "(unknown)"
        log.warning(f"  [DIAG] Default source: {default_src}")

        # Source state
        r = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 5:
                    name, state = parts[1], parts[4]
                    if "bluez" in name or name == default_src:
                        log.warning(f"  [DIAG] Source: {name} → {state}")

        # Default sink (output switch can break BT profile)
        r = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=3)
        default_sink = r.stdout.strip() if r.returncode == 0 else "(unknown)"
        is_bt_sink = "bluez" in default_sink
        is_hdmi_sink = "hdmi" in default_sink.lower()
        log.warning(f"  [DIAG] Default sink: {default_sink}")
        if is_hdmi_sink and "bluez" in default_src:
            log.warning(f"  [DIAG] ⚠ BT mic selected but output on HDMI — HFP profile likely broken")
        elif not is_bt_sink and "bluez" in default_src:
            log.warning(f"  [DIAG] ⚠ BT mic selected but output not on BT — profile mismatch possible")

        # BT connection state via bluetoothctl
        if "bluez" in default_src:
            # Extract MAC from source name like bluez_input.50:C2:ED:10:A7:43
            mac = default_src.replace("bluez_input.", "").replace("_", ":")
            r = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if any(k in line for k in ("Name:", "Connected:", "UUID: Handsfree", "UUID: Audio")):
                        log.warning(f"  [DIAG] BT: {line}")
    except Exception as ex:
        log.debug(f"  [DIAG] diagnostics failed: {ex}")


def _try_bt_reconnect() -> bool:
    """Attempt to fix a dead BT mic by disconnecting and reconnecting.

    Returns True if mic is ready after reconnect.
    """
    try:
        r = subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True, timeout=3)
        default_src = r.stdout.strip() if r.returncode == 0 else ""
        if "bluez" not in default_src:
            log.info("  [2] Not a BT mic — skipping reconnect")
            return False

        mac = default_src.replace("bluez_input.", "").replace("_", ":")
        log.warning(f"  [2] Attempting BT reconnect for {mac}...")

        r = subprocess.run(["bluetoothctl", "disconnect", mac],
                           capture_output=True, text=True, timeout=5)
        log.info(f"  [2] BT disconnect: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.returncode}")

        time.sleep(2)

        r = subprocess.run(["bluetoothctl", "connect", mac],
                           capture_output=True, text=True, timeout=10)
        log.info(f"  [2] BT connect: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.returncode}")

        time.sleep(3)

        _close_audio_stream("bt-reconnect")
        _open_audio_stream()
        mic_ready = _check_mic_ready(max_wait_s=5.0)
        log.info(f"  [2] BT reconnect result: mic_ready={mic_ready}")
        return mic_ready

    except Exception as ex:
        log.warning(f"  [2] BT reconnect failed: {ex}")
        return False


def _read_chunk(stream, n_samples: int, timeout: float = 5.0) -> bytes:
    """Read n_samples of 16kHz mono int16 from parec stdout.

    Uses select() to avoid blocking forever when the audio device stops
    delivering data (e.g. Bluetooth mic disconnects while parec is running).
    """
    n_bytes = n_samples * 2  # int16 = 2 bytes per sample
    fd = stream.stdout.fileno()
    buf = b""
    deadline = time.time() + timeout
    while len(buf) < n_bytes:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise IOError(f"Audio read timeout ({timeout}s) — mic not delivering data")
        ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
        if not ready:
            continue
        chunk = os.read(fd, n_bytes - len(buf))
        if not chunk:
            raise IOError("Audio stream ended (parec terminated)")
        buf += chunk
    return buf


def _open_audio_stream():
    """Opens 16kHz mono input stream via parec (PulseAudio/PipeWire native)."""
    global _audio_proc, _audio_stream

    if _audio_proc is not None and _audio_proc.poll() is None:
        return  # already running

    cmd = [
        "parec",
        "--rate", str(SAMPLE_RATE),
        "--channels", "1",
        "--format", "s16le",
        "--latency-msec", "30",
    ]
    if INPUT_DEVICE:
        cmd += ["--device", INPUT_DEVICE]

    log.info(f"  Starting audio: {' '.join(cmd)}")
    _audio_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _audio_stream = _audio_proc
    log.info(f"  ✅ Input stream opened | {_mic_device_info()}")


def _close_audio_stream(reason: str = "idle-timeout"):
    """Terminates the parec subprocess."""
    global _audio_proc, _audio_stream

    if _audio_proc is None:
        return

    try:
        _audio_proc.terminate()
        _audio_proc.wait(timeout=2)
    except Exception as ex:
        log.warning(f"  ⚠️  close_stream failed ({reason}): {ex}")
        try:
            _audio_proc.kill()
        except Exception:
            pass
    finally:
        _audio_proc = None
        _audio_stream = None
        log.info(f"  💤 Input stream closed ({reason})")


def _cancel_idle_close_timer():
    global _idle_close_timer
    with _idle_timer_lock:
        if _idle_close_timer is not None:
            _idle_close_timer.cancel()
            _idle_close_timer = None


def _schedule_idle_close(seconds: int = IDLE_CLOSE_SECONDS, reason: str = "idle-timeout"):
    """Schedules full stream close after inactivity."""
    global _idle_close_timer

    _cancel_idle_close_timer()

    def _idle_close_task():
        global _idle_close_timer
        with _idle_timer_lock:
            _idle_close_timer = None
        # Defer close while recording or while watch loop is actively reading.
        # Closing the stream while the watch loop blocks in read() causes a
        # PortAudio segfault that kills the process.
        if _recording_lock.locked() or (WATCH_ACTIVE and not WATCH_SUSPENDED):
            _schedule_idle_close(seconds=seconds, reason=reason)
            return
        _close_audio_stream(reason=reason)

    with _idle_timer_lock:
        _idle_close_timer = threading.Timer(seconds, _idle_close_task)
        _idle_close_timer.daemon = True
        _idle_close_timer.start()
    log.info(f"  ⏳ Idle close scheduled in {seconds}s ({reason})")


def _build_trigger_prompt_terms():
    terms = []
    for trig in ACTION_TRIGGERS:
        for phrase in trig.get("match", []):
            p = str(phrase).strip()
            if p:
                terms.append(p)
    # stable unique order
    seen = set()
    uniq = []
    for t in terms:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq


def _apply_auto_prompt_from_triggers():
    """Inject trigger keywords into INITIAL_PROMPT so ASR is biased towards command phrases."""
    global INITIAL_PROMPT
    if not AUTO_PROMPT_FROM_TRIGGERS:
        return
    terms = _build_trigger_prompt_terms()
    if not terms:
        return

    auto_prompt = "voice commands: " + ", ".join(terms)
    if INITIAL_PROMPT and str(INITIAL_PROMPT).strip():
        INITIAL_PROMPT = f"{INITIAL_PROMPT.strip()} | {auto_prompt}"
    else:
        INITIAL_PROMPT = auto_prompt
    log.info(f"  ⚙️  prompt auto-augmented from {len(terms)} trigger keywords")


def _parse_duration_to_seconds(raw):
    """Parses strings like 5m, 60s, 1h into seconds."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    units = {'s': 1, 'm': 60, 'h': 3600}
    unit = s[-1]
    if unit in units:
        try:
            return int(float(s[:-1]) * units[unit])
        except ValueError:
            return None
    return None


def _watch_set_duration(duration_raw):
    global WATCH_UNTIL_TS, WATCH_ACTIVE
    sec = _parse_duration_to_seconds(duration_raw)
    if sec is None or sec < 0:
        return False, f"invalid duration: {duration_raw}"
    WATCH_ACTIVE = sec > 0
    WATCH_UNTIL_TS = (time.time() + sec) if sec > 0 else None
    return True, f"watch duration set/extended by {duration_raw}"


def _match_trigger(text_lower, source="record"):
    for trig in ACTION_TRIGGERS:
        name = trig.get("name", "")

        # Context-aware matching for active-window typing (verb + location marker).
        if name == "type_to_active":
            allow_context = not (CONTEXT_MATCH_WATCH_ONLY and source != "watch")
            if allow_context:
                _verb_pat = r"\b(?:type|write|tippe?|schreib(?:e)?|diktier(?:e)?)\b"
                _loc_pat  = r"\b(?:hier|here|direkt|aktiv\w*|fenster|window)\b"
                if REQUIRE_COMMAND_PREFIX and source != "watch":
                    has_verb = bool(re.search(r"^\s*" + _verb_pat, text_lower))
                else:
                    has_verb = bool(re.search(_verb_pat, text_lower))
                has_loc = bool(re.search(_loc_pat, text_lower))
                if has_verb and has_loc:
                    return trig, "__active_intent__"

        # Context-aware matching for telegram action (English + German verbs).
        if name == "type_to_telegram":
            allow_context = not (CONTEXT_MATCH_WATCH_ONLY and source != "watch")
            if allow_context:
                has_target = bool(re.search(r"\btelegramm?\b", text_lower))
                # German: sende/schicke/schreibe/übertrage/übermittle + English: send/type/write/open
                _verb_pat = r"\b(?:send|type|write|go\s+to|open|sende|schick(?:e|en)?|übertrag(?:e|en)?|übermittle|schreib(?:e)?)\b"
                # In watch mode verb may appear anywhere; in record mode respect REQUIRE_COMMAND_PREFIX.
                if REQUIRE_COMMAND_PREFIX and source != "watch":
                    has_verb = bool(re.search(r"^\s*" + _verb_pat, text_lower))
                else:
                    has_verb = bool(re.search(_verb_pat, text_lower))
                if has_target and has_verb:
                    return trig, "__telegram_intent__"

        for phrase in trig.get("match", []):
            p = str(phrase).strip().lower()
            if p and p in text_lower:
                return trig, p
    return None, None


def _run_action_trigger(trig, text, raw_text=None):
    command = trig.get("command")
    args = [str(a) for a in trig.get("args", [])]
    if not command:
        return False, "trigger has no command"

    env = os.environ.copy()
    env["OKAWISP_TEXT"] = text or ""
    env["OKAWISP_TEXT_RAW"] = raw_text if raw_text is not None else (text or "")
    env["OKAWISP_TRIGGER"] = trig.get("name", "")

    min_chars = trig.get("min_chars")
    if isinstance(min_chars, int) and len((text or "").strip()) < min_chars:
        return False, f"min_chars not reached ({min_chars})"

    try:
        proc = subprocess.run([command, *args], env=env, capture_output=True, text=True, timeout=20)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if out:
            log.info(f"  [action] stdout: {out}")
        if err:
            log.warning(f"  [action] stderr: {err}")
        if proc.returncode != 0:
            return False, f"exit={proc.returncode}"
        return True, "ok"
    except Exception as ex:
        return False, str(ex)


def _build_status_payload():
    now = time.time()
    watch_remaining = None
    if WATCH_UNTIL_TS:
        watch_remaining = max(0, int(WATCH_UNTIL_TS - now))
    return {
        "ok": True,
        "pid": os.getpid(),
        "recording": _recording_lock.locked(),
        "watch_active": WATCH_ACTIVE,
        "watch_remaining_s": watch_remaining,
        "idle_auto_close_seconds": IDLE_CLOSE_SECONDS,
        "control_socket": CONTROL_SOCKET_PATH,
    }


def _handle_control_request(req):
    global IDLE_CLOSE_SECONDS, WATCH_ACTIVE, WATCH_UNTIL_TS

    op = req.get("op")
    if op == "status":
        return _build_status_payload()

    if op == "watch.start":
        WATCH_ACTIVE = True
        WATCH_UNTIL_TS = None
        return {"ok": True, "message": "watch started"}

    if op == "watch.stop":
        WATCH_ACTIVE = False
        WATCH_UNTIL_TS = None
        # Schedule stream close shortly — safe now that watch loop will idle.
        _schedule_idle_close(3, "watch-stop")
        return {"ok": True, "message": "watch stopped"}

    if op == "watch.duration":
        ok, msg = _watch_set_duration(req.get("duration"))
        return {"ok": ok, "message": msg}

    if op == "watch.set_idle_close":
        sec = _parse_duration_to_seconds(req.get("idle_close"))
        if sec is None or sec < 0:
            return {"ok": False, "message": "invalid idle_close"}
        IDLE_CLOSE_SECONDS = sec
        return {"ok": True, "message": f"idle_auto_close_seconds={sec}"}

    if op == "test.trigger":
        name = req.get("name")
        text = req.get("text", "")
        trig = None
        for t in ACTION_TRIGGERS:
            if t.get("name") == name:
                trig = t
                break
        if trig is None:
            return {"ok": False, "message": f"trigger not found: {name}"}
        ok, msg = _run_action_trigger(trig, text, raw_text=text)
        return {"ok": ok, "message": msg, "trigger": name}

    return {"ok": False, "message": f"unknown op: {op}"}


def _control_server_loop():
    os.makedirs(os.path.dirname(CONTROL_SOCKET_PATH), exist_ok=True)
    if os.path.exists(CONTROL_SOCKET_PATH):
        try:
            os.remove(CONTROL_SOCKET_PATH)
        except Exception:
            pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(CONTROL_SOCKET_PATH)
    os.chmod(CONTROL_SOCKET_PATH, 0o600)
    srv.listen(16)
    srv.settimeout(1.0)

    log.info(f"  🔌 Control socket listening: {CONTROL_SOCKET_PATH}")

    def _handle_conn(conn):
        with conn:
            try:
                data = conn.recv(65536)
                req = json.loads(data.decode("utf-8", errors="ignore") or "{}")
                resp = _handle_control_request(req)
            except Exception as ex:
                resp = {"ok": False, "message": str(ex)}
            try:
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except Exception:
                pass

    try:
        while not should_exit:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except Exception:
                continue
            threading.Thread(target=_handle_conn, args=(conn,), daemon=True).start()
    finally:
        try:
            srv.close()
        except Exception:
            pass
        try:
            if os.path.exists(CONTROL_SOCKET_PATH):
                os.remove(CONTROL_SOCKET_PATH)
        except Exception:
            pass


def _start_control_server():
    threading.Thread(target=_control_server_loop, daemon=True).start()


def _watch_transcription_worker():
    """Background thread: dequeues audio segments, transcribes, matches triggers."""
    while not should_exit:
        try:
            audio_np = _watch_transcribe_queue.get(timeout=1.0)
        except _queue_mod.Empty:
            continue

        # Transcribe
        try:
            if engine_type == "api":
                text, _lang = transcribe_via_api(audio_np)
            elif engine_type == "faster":
                text, _lang = transcribe_faster_whisper(audio_np)
            elif model is not None:
                text, _lang = transcribe_openai_whisper(audio_np)
            else:
                text = ""
        except Exception as ex:
            log.warning(f"  [watch-worker] transcribe failed: {ex}")
            continue

        text = (text or "").strip()
        if not text:
            continue

        duration_s = len(audio_np) / SAMPLE_RATE
        log.info(f"  [watch] transcribed ({duration_s:.1f}s): {text[:180]}")

        # Match trigger — single-segment, no accumulation
        trig, phrase = _match_trigger(text.lower(), source="watch")
        if trig is None:
            continue

        name = trig.get("name", "<unnamed>")
        log.info(f"  [watch] trigger matched: '{name}' (phrase='{phrase}')")

        # Build payload: strip the command phrase from the text
        payload = text
        if name == "type_to_active":
            payload = re.sub(_PAT_TYPE_TO_ACTIVE, "", text, count=1,
                             flags=re.IGNORECASE).strip(" ,.!?;:-")
        elif name == "type_to_telegram":
            payload = re.sub(_PAT_TYPE_TO_TELEGRAM, "", text, count=1,
                             flags=re.IGNORECASE).strip(" ,.!?;:-")
        else:
            if phrase:
                idx = text.lower().find(phrase)
                if idx >= 0:
                    payload = (text[:idx] + text[idx + len(phrase):]).strip(" ,.!?;:-")

        # Typing triggers need actual payload text
        if not payload and name in ("type_to_telegram", "type_to_active"):
            log.info(f"  [watch] trigger '{name}' skipped (empty payload)")
            continue
        if not payload:
            payload = text

        # Cooldown check
        now = time.time()
        last = _trigger_last_fire.get(name, 0)
        if (now - last) * 1000 < ACTION_COOLDOWN_MS:
            log.debug(f"  [watch] trigger '{name}' cooldown active")
            continue
        _trigger_last_fire[name] = now

        # Execute
        ok, msg = _run_action_trigger(trig, payload, raw_text=text)
        if ok:
            log.info(f"  [watch] action OK: {name} | '{payload[:120]}'")
        else:
            log.warning(f"  [watch] action FAIL: {name}: {msg}")


def _watch_loop():
    """Background watch loop: VAD-based speech segmentation, non-blocking transcription.

    Reads 512-sample chunks (32ms), runs silero-vad to detect speech boundaries.
    When a speech segment ends (VAD silence >= WATCH_SILENCE_MS), the audio
    is enqueued for background transcription. The watch loop never blocks on
    Whisper — it keeps reading audio continuously.
    """
    global WATCH_ACTIVE, WATCH_UNTIL_TS, _audio_stream
    import torch

    vad_chunk = VAD_CHUNK_SAMPLES  # 512 samples = 32ms
    silence_chunks_needed = max(1, int(WATCH_SILENCE_MS / 32))
    max_segment_chunks = max(1, int(WATCH_MAX_SEGMENT_MS / 32))
    min_segment_chunks = max(1, int(WATCH_MIN_SEGMENT_MS / 32))

    # Start transcription worker thread
    threading.Thread(target=_watch_transcription_worker, daemon=True).start()

    # Segment state
    in_segment = False
    segment_frames = []
    segment_chunk_count = 0
    silence_streak = 0

    while not should_exit:
        now = time.time()

        # Duration expiry
        if WATCH_UNTIL_TS and now >= WATCH_UNTIL_TS:
            WATCH_ACTIVE = False
            WATCH_UNTIL_TS = None

        # Suspend/inactive: discard partial segment, sleep briefly
        if (not WATCH_ACTIVE) or _recording_lock.locked() or WATCH_SUSPENDED:
            if in_segment:
                in_segment = False
                segment_frames = []
                segment_chunk_count = 0
                silence_streak = 0
                log.debug("  [watch] segment discarded (suspended)")
            time.sleep(0.05)
            continue

        # Ensure stream is open and active
        try:
            _cancel_idle_close_timer()
            if _audio_proc is not None and _audio_proc.poll() is not None:
                _close_audio_stream("watch-proc-dead")
            _open_audio_stream()

            data = _read_chunk(_audio_stream, vad_chunk)
        except Exception as ex:
            msg = str(ex)
            if "stream ended" in msg.lower() or "terminated" in msg.lower():
                _close_audio_stream("watch-stream-error")
                log.warning("  [watch] stream error -> recovering")
                time.sleep(0.35)
                continue
            log.warning(f"  [watch] read error: {ex}")
            time.sleep(0.2)
            continue

        # VAD inference (silero-vad, < 1ms on CPU)
        audio_f32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        with torch.no_grad():
            speech_prob = _vad_model(torch.from_numpy(audio_f32), SAMPLE_RATE).item()
        is_speech = speech_prob >= VAD_THRESHOLD

        if is_speech:
            silence_streak = 0
            if not in_segment:
                in_segment = True
                segment_frames = []
                segment_chunk_count = 0
                log.debug(f"  [watch] speech start (prob={speech_prob:.2f})")
            segment_frames.append(data)
            segment_chunk_count += 1
        elif in_segment:
            # Keep silence tail for Whisper context
            segment_frames.append(data)
            segment_chunk_count += 1
            silence_streak += 1

        # Check segment end conditions
        segment_done = False
        if in_segment:
            if silence_streak >= silence_chunks_needed:
                segment_done = True
                log.debug(f"  [watch] speech end (silence {silence_streak * 32}ms)")
            elif segment_chunk_count >= max_segment_chunks:
                segment_done = True
                log.debug(f"  [watch] speech end (max duration {WATCH_MAX_SEGMENT_MS}ms)")

        if segment_done:
            in_segment = False
            silence_streak = 0

            # Skip segments that are too short (noise bursts)
            if segment_chunk_count < min_segment_chunks:
                log.debug(f"  [watch] segment too short ({segment_chunk_count * 32}ms), discarded")
                segment_frames = []
                segment_chunk_count = 0
                continue

            # Convert to float32 and enqueue for background transcription
            audio_data = b"".join(segment_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            segment_frames = []
            segment_chunk_count = 0

            try:
                _watch_transcribe_queue.put_nowait(audio_np)
            except _queue_mod.Full:
                log.warning("  [watch] transcription queue full, segment dropped")

            # Reset VAD state after segment to avoid LSTM state leakage
            try:
                _vad_model.reset_states()
            except Exception:
                pass


def _start_watch_thread():
    threading.Thread(target=_watch_loop, daemon=True).start()


def do_voice_input(ptt_mode: bool = False, skip_start_sound: bool = False):
    """Main function: Record → Transcribe → Type

    ptt_mode: If True, Push-to-Talk mode (stop on key release, send with Enter)
    skip_start_sound: If True, start sound was already played (PTT mode)
    """
    global _ptt_stop_requested, WATCH_SUSPENDED, WATCH_ACTIVE

    # Lock instead of bool: prevents race condition on rapid hotkey presses
    if not _recording_lock.acquire(blocking=False):
        log.debug("Hotkey ignored — recording already running (lock held)")
        return

    # Reset PTT stop flag
    _ptt_stop_requested = False
    _recording_ready_event.clear()
    _recording_stopped_event.clear()
    WATCH_SUSPENDED = True

    mode_str = "PTT" if ptt_mode else "Toggle"
    log.info("━" * 60)
    log.info(f"▶  Recording sequence starting ({mode_str} mode)")
    try:
        _run_voice_input(ptt_mode=ptt_mode, skip_start_sound=skip_start_sound)
    finally:
        _ptt_stop_requested = False
        WATCH_SUSPENDED = False
        # Key press always re-enables watch mode if it was disabled.
        # (Triggers only fire in watch mode, not during key recordings.)
        if not WATCH_ACTIVE:
            WATCH_ACTIVE = True
            WATCH_UNTIL_TS = None
            log.info("  🎙️  Watch mode re-enabled by key press")
        _recording_lock.release()
        _schedule_idle_close(seconds=IDLE_CLOSE_SECONDS, reason="post-record-idle")
        log.info("◀  Recording sequence finished")
        log.info("━" * 60)


def _run_voice_input(ptt_mode: bool = False, skip_start_sound: bool = False):
    """Actual recording logic (runs under _recording_lock).
    
    ptt_mode: If True, PTT mode - stop on key release, send with Enter at end.
    skip_start_sound: If True, start sound was already played externally.
    """
    global _audio_proc, _audio_stream

    _cancel_idle_close_timer()
    # Fresh parec process for recording (avoids pipe race with watch loop)
    _close_audio_stream("recording-start")
    _open_audio_stream()

    # ── 1. Check stream status ─────────────────────────────────────
    log.info(f"  [1] Input device: {_mic_device_info()}")

    if _audio_proc is None or _audio_proc.poll() is not None:
        log.error("  [1] ❌ parec failed to start")
        notify("❌ Audio error", "parec not running", "critical")
        return

    # ── 2. Device readiness check (wait for Bluetooth reconnect) ───
    log.info(f"  [2] Checking if microphone delivers real data...")
    mic_ready = _check_mic_ready(max_wait_s=3.0)
    log.info(f"  [2] Mic-Ready: {mic_ready} | Device: {_mic_device_info()}")

    if not mic_ready:
        log.warning("  [2] Mic not ready — attempting stream reopen...")
        try:
            _close_audio_stream("mic-not-ready")
            _open_audio_stream()
            log.info("  [2] Stream reopened — second readiness check...")
            mic_ready = _check_mic_ready(max_wait_s=2.0)
            log.info(f"  [2] Second check: mic_ready={mic_ready}")
        except Exception as e:
            log.error(f"  [2] ❌ Stream restart failed: {e}")
            notify("❌ Mic not ready", f"Mic error: {e}", "critical")
            play_sound("mic_error")
            return

    if not mic_ready:
        # Try BT reconnect as last resort
        mic_ready = _try_bt_reconnect()

    if not mic_ready:
        log.error("  [2] ❌ Mic not ready after all retries — aborting")
        notify("❌ Mic not ready", "Microphone not available", "critical")
        play_sound("mic_error")
        return

    # Signal that recording pipeline is armed and mic is ready.
    _recording_ready_event.set()

    # ── 2b. Start-Sound ───────────────────────────────────────────
    if not skip_start_sound:
        play_sound("record_start")
        notify("🔴 Recording", "Speak now...")
    else:
        notify("🔴 PTT Recording", "Speak...")

    # ── 2c. Drain stale buffer after start sound ──────────────────
    # Read and discard ~0.5s of buffered audio to get fresh data.
    drain_count = max(1, int(0.5 * SAMPLE_RATE / CHUNK_SIZE))
    for _ in range(drain_count):
        try:
            _read_chunk(_audio_stream, CHUNK_SIZE)
        except Exception:
            break

    # ── 2d. Calibration (RMS fallback only) ──────────────────────
    use_vad = VAD_ENABLED and _vad_model is not None
    if not use_vad:
        dynamic_threshold = calibrate_silence_threshold(_audio_stream)
        log.info(f"  [2d] RMS-Fallback: threshold={dynamic_threshold}")
    else:
        dynamic_threshold = None

    if use_vad:
        log.info(f"  [3] 🔴 Recording (VAD, threshold={VAD_THRESHOLD}, "
                 f"min_silence={VAD_MIN_SILENCE_MS}ms)")
    else:
        log.info(f"  [3] 🔴 Recording (RMS, Silence-Limit={SILENCE_DURATION}s, "
                 f"Threshold={dynamic_threshold})")

    # ── 4. Recording ─────────────────────────────────────────────
    record_start_ts = time.time()
    try:
        if use_vad:
            frames = record_with_vad(_audio_stream, ptt_mode=ptt_mode)
        else:
            frames = record_with_silence_detection(_audio_stream, threshold=dynamic_threshold)
    except Exception as e:
        log.error(f"  [4] ❌ Recording-Error: {e}")
        notify("❌ Error", str(e), "critical")
        frames = []
    finally:
        elapsed_record = time.time() - record_start_ts
        _close_audio_stream("recording-end")
        log.info(f"  [4] ⏹  Stream closed after {elapsed_record:.2f}s")
        _recording_stopped_event.set()

    # ── 5. Stop-Sound ───────────────────────────────────────────────
    if not ptt_mode:
        play_sound("record_end")
    time.sleep(0.3)

    if not frames:
        log.warning("  [5] No audio frames recorded")
        notify("⚠️ No recording", "No audio data")
        return

    # ── 6. Analyze audio ────────────────────────────────────────
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio_np) / SAMPLE_RATE
    rms_overall = float(np.sqrt(np.mean(audio_np ** 2))) * 32768
    log.info(f"  [6] Audio: {duration:.2f}s, {len(frames)} chunks, RMS={rms_overall:.1f}")

    if duration < 0.5:
        log.warning("  [6] Recording too short (<0.5s) — discarded")
        return

    # ── 7. Transcription ─────────────────────────────────────────
    log.info(f"  [7] Transcribing ({duration:.1f}s Audio, engine={engine_type})...")
    t0 = time.time()
    try:
        if engine_type == "api":
            text, lang = transcribe_via_api(audio_np)
        elif engine_type == "faster":
            text, lang = transcribe_faster_whisper(audio_np)
        else:
            text, lang = transcribe_openai_whisper(audio_np)
        elapsed_trans = time.time() - t0
    except Exception as e:
        log.error(f"  [7] ❌ Transcription error: {e}")
        notify("❌ Error", str(e), "critical")
        return

    if not text:
        log.warning(f"  [7] No speech detected (duration={duration:.1f}s, RMS={rms_overall:.1f})")
        notify("⚠️ No text", "No speech detected")
        return

    log.info(f"  [7] 📝 [{lang}] ({elapsed_trans:.1f}s) {text}")

    # ── 7b. Action triggers (watch-only) ───────────────────────
    # Triggers only fire in watch mode, not during key-press recordings.
    # During PTT/key recording the full text is typed normally.

    # ── 8. Type text into active window ─────────────────────────
    time.sleep(0.3)
    typed = type_text(text)
    if typed:
        notify("✅ Typed", text[:80])
        log.info("  [8] ✅ Text typed into window")
        
        # PTT mode: press Enter to send
        if ptt_mode:
            time.sleep(0.1)
            try:
                subprocess.run(['xdotool', 'key', 'Return'], check=True, timeout=2)
                log.info("  [8] ⏎  PTT mode: Enter pressed (message sent)")
            except Exception as e:
                log.warning(f"  [8] ⚠️  PTT Enter failed: {e}")
    else:
        notify("⚠️ Error", "Typing failed (xdotool)", urgency="critical")
        log.error("  [8] ❌ xdotool typing failed")


def listen_keyboard_hotkey(hotkey):
    """Listen for global hotkey via pynput"""
    try:
        from pynput import keyboard

        alt_gr_key = getattr(keyboard.Key, 'alt_gr', None) or getattr(keyboard.Key, 'alt_r', None)

        key_map = {
            'F1': keyboard.Key.f1, 'F2': keyboard.Key.f2,
            'F3': keyboard.Key.f3, 'F4': keyboard.Key.f4,
            'F5': keyboard.Key.f5, 'F6': keyboard.Key.f6,
            'F7': keyboard.Key.f7, 'F8': keyboard.Key.f8,
            'F9': keyboard.Key.f9, 'F10': keyboard.Key.f10,
            'F11': keyboard.Key.f11, 'F12': keyboard.Key.f12,
        }

        # AltGr aliases (layout-dependent naming)
        if alt_gr_key is not None:
            key_map.update({
                'ALT_GR': alt_gr_key,
                'ALTGR': alt_gr_key,
                'RIGHT_ALT': alt_gr_key,
                'RALT': alt_gr_key,
            })

        hotkey_norm = hotkey.upper().replace('-', '_').replace(' ', '_')
        target_key = key_map.get(hotkey_norm)
        
        # AltGr special handling: on Linux/X11 systems AltGr can be various keysyms
        # See: /usr/include/X11/keysymdef.h
        altgr_vk_codes = {
            65027,  # 0xFE03 = XK_ISO_Level3_Shift (AltGr on European ISO keyboards: DE, CH, FR, etc.)
            65406,  # 0xFF7E = XK_Alt_R (Right Alt on US keyboards)
            65312,  # 0xFF20 = XK_Multi_key (Compose, sometimes used)
            65511,  # 0xFFE7 = XK_Meta_L (rare)
            65512,  # 0xFFE8 = XK_Meta_R (rare)
            65513,  # 0xFFE9 = XK_Alt_L
            65514,  # 0xFFEA = XK_Alt_R (alternative)
        }
        is_altgr_hotkey = hotkey_norm in ('ALT_GR', 'ALTGR', 'RIGHT_ALT', 'RALT')
        
        if not target_key and not is_altgr_hotkey:
            print(f"  ❌ Unknown hotkey: {hotkey}")
            print(f"     Available: {', '.join(sorted(set(key_map.keys())))}")
            sys.exit(1)

        # PTT (Push-to-Talk) state
        ptt_state = {
            'press_time': None,
            'ptt_active': False,
            'ptt_threshold': 0.3,  # 300ms to trigger PTT mode
        }

        def is_hotkey(key):
            """Check if key matches configured hotkey"""
            if target_key and key == target_key:
                return True
            if is_altgr_hotkey and hasattr(key, 'vk') and key.vk in altgr_vk_codes:
                return True
            return False

        def on_press(key):
            if not is_hotkey(key):
                return
            
            # Debug: Log every hotkey press
            log.debug(f"🔑 Hotkey PRESS detected: {key}")
            
            # Already recording? Ignore
            if _recording_lock.locked():
                log.debug("🔑 Ignored (recording in progress)")
                return
            
            # Record press time for PTT detection
            if ptt_state['press_time'] is None:
                ptt_state['press_time'] = time.time()
                ptt_state['ptt_active'] = False
                
                # Start delayed PTT check
                def check_ptt():
                    time.sleep(ptt_state['ptt_threshold'])
                    if ptt_state['press_time'] is not None and not _recording_lock.locked():
                        # Still held after threshold → PTT mode
                        ptt_state['ptt_active'] = True
                        log.info("▶  PTT mode — Recording while key held")

                        # Start recording FIRST to avoid clipping first spoken words.
                        thread = threading.Thread(
                            target=lambda: do_voice_input(ptt_mode=True, skip_start_sound=True),
                            daemon=True,
                        )
                        thread.start()

                        # Optional UX chime after recording has started (non-blocking).
                        def _late_ptt_chime():
                            # Event-driven: chime only after recording path reports mic-ready.
                            # Timeout keeps UX responsive if readiness event is delayed.
                            ready = _recording_ready_event.wait(timeout=1.5)
                            if not ready:
                                log.warning("  [ptt] ready-event timeout; playing start chime fallback")
                            play_sound("record_start")
                        threading.Thread(target=_late_ptt_chime, daemon=True).start()
                
                threading.Thread(target=check_ptt, daemon=True).start()

        def on_release(key):
            if not is_hotkey(key):
                return
            
            # Debug: Log every hotkey release
            log.debug(f"🔑 Hotkey RELEASE detected: {key}")
            
            press_duration = time.time() - ptt_state['press_time'] if ptt_state['press_time'] else 0
            was_ptt = ptt_state['ptt_active']
            
            # Reset state
            ptt_state['press_time'] = None
            ptt_state['ptt_active'] = False
            
            if was_ptt:
                # PTT mode: request stop, then play stop sound when stream actually stopped.
                log.info(f"◀  PTT release after {press_duration:.1f}s — Stopping")
                global _ptt_stop_requested
                _ptt_stop_requested = True

                def _ptt_stop_chime_on_event():
                    stopped = _recording_stopped_event.wait(timeout=2.5)
                    if not stopped:
                        log.warning("  [ptt] stop-event timeout; playing stop chime fallback")
                    play_sound("record_end")

                threading.Thread(target=_ptt_stop_chime_on_event, daemon=True).start()
            elif press_duration < ptt_state['ptt_threshold'] and not _recording_lock.locked():
                # Short press: toggle mode (original behavior)
                log.info("▶  Short press — Toggle mode")
                thread = threading.Thread(target=lambda: do_voice_input(ptt_mode=False), daemon=True)
                thread.start()

        print(f"  🎹 Hotkey: {hotkey}")
        print(f"  📍 Method: pynput (global keyboard listener)")
        print(f"  📍 PTT mode: hold key ≥300ms = Push-to-Talk")

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    except ImportError:
        return False

    return True


def listen_xbindkeys_hotkey(hotkey):
    """Fallback: Enter-based input"""
    print(f"  ⚠️  pynput not installed.")
    print(f"  Install: pip install pynput")
    print()

    while not should_exit:
        try:
            input(f"  ⏎  Press ENTER to record (Ctrl+C to exit)... ")
            thread = threading.Thread(target=do_voice_input, daemon=True)
            thread.start()
            thread.join()
        except (EOFError, KeyboardInterrupt):
            break


def main():
    global model, should_exit, engine_type
    global MODEL_SIZE, LANGUAGE, ENGINE, BEAM_SIZE, INITIAL_PROMPT
    global SILENCE_DURATION, SILENCE_THRESHOLD
    global VAD_ENABLED, VAD_THRESHOLD, VAD_MIN_SILENCE_MS
    global INPUT_DEVICE
    global CUSTOM_RECORD_START_SOUND, CUSTOM_RECORD_END_SOUND
    global WHISPER_API_BASE_URL, WHISPER_API_KEY, WHISPER_API_MODEL
    global IDLE_CLOSE_SECONDS, ACTION_TRIGGERS, ACTION_COOLDOWN_MS, AUTO_PROMPT_FROM_TRIGGERS, WATCH_ACTIVE
    global WATCH_MAX_SEGMENT_MS, WATCH_SILENCE_MS, WATCH_MIN_SEGMENT_MS

    # ── Load config file (defaults) ──────────────────────────────
    cfg = load_config()
    rec = cfg.get("recording", {})
    vad = cfg.get("vad", {})
    sounds = cfg.get("sounds", {})
    api_cfg = cfg.get("api", {})
    watch_cfg = cfg.get("watch", {})
    actions_cfg = cfg.get("actions", {})

    # Input device override from [recording] section
    # Value is a PulseAudio source name (e.g. "bluez_input.XX:XX:XX:XX:XX:XX")
    if rec.get("device") is not None:
        INPUT_DEVICE = str(rec["device"])
        log.info(f"  Config: device override = {INPUT_DEVICE}")

    # Config values as defaults (CLI overrides these)
    if vad.get("enabled") is not None:    VAD_ENABLED        = vad["enabled"]
    if vad.get("threshold") is not None:  VAD_THRESHOLD      = float(vad["threshold"])
    if vad.get("min_silence_ms"):         VAD_MIN_SILENCE_MS = int(vad["min_silence_ms"])
    if sounds.get("start"):               CUSTOM_RECORD_START_SOUND = sounds["start"]
    if sounds.get("stop"):                CUSTOM_RECORD_END_SOUND   = sounds["stop"]
    if api_cfg.get("base_url"):           WHISPER_API_BASE_URL = api_cfg["base_url"]
    if api_cfg.get("api_key"):            WHISPER_API_KEY      = api_cfg["api_key"]
    if api_cfg.get("model"):              WHISPER_API_MODEL    = api_cfg["model"]

    if watch_cfg.get("idle_auto_close_seconds") is not None:
        try:
            IDLE_CLOSE_SECONDS = int(watch_cfg["idle_auto_close_seconds"])
        except Exception:
            pass
    WATCH_ACTIVE = IDLE_CLOSE_SECONDS > 0

    if watch_cfg.get("max_segment_ms") is not None:
        try:
            WATCH_MAX_SEGMENT_MS = int(watch_cfg["max_segment_ms"])
        except Exception:
            pass
    if watch_cfg.get("silence_ms") is not None:
        try:
            WATCH_SILENCE_MS = int(watch_cfg["silence_ms"])
        except Exception:
            pass
    if watch_cfg.get("min_segment_ms") is not None:
        try:
            WATCH_MIN_SEGMENT_MS = int(watch_cfg["min_segment_ms"])
        except Exception:
            pass

    if actions_cfg.get("action_cooldown_ms") is not None:
        try:
            ACTION_COOLDOWN_MS = int(actions_cfg["action_cooldown_ms"])
        except Exception:
            pass
    if actions_cfg.get("auto_prompt_from_triggers") is not None:
        AUTO_PROMPT_FROM_TRIGGERS = bool(actions_cfg.get("auto_prompt_from_triggers"))

    ACTION_TRIGGERS = actions_cfg.get("triggers", []) if isinstance(actions_cfg.get("triggers", []), list) else []

    parser = argparse.ArgumentParser(
        description='OkaWhisp - System-Level Voice Input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python okawhisp.py                                       # Default (AltGr, medium, German)
  python okawhisp.py --model small --language en           # English, faster
  python okawhisp.py --engine api --api-key sk-...         # OpenAI API
  python okawhisp.py --engine api \\
    --api-url https://api.groq.com/openai/v1 --api-key gsk_  # Groq (free, fast)
  python okawhisp.py --prompt "NestJS, Flutter, API"        # Technical terms as context
  python okawhisp.py --beam-size 1                          # Faster, slightly less accurate

Config file: ~/.config/okawhisp/config.toml
        """
    )
    parser.add_argument('--key', default=rec.get('key', 'ALT_GR'),
                        help='Recording hotkey (ALT_GR or F1-F12, default: ALT_GR)')
    parser.add_argument('--model', default=rec.get('model', 'medium'),
                        help='Whisper Model: tiny/base/small/medium/large-v3 (default: medium)')
    parser.add_argument('--language', default=rec.get('language', 'de'),
                        help='Language: de/en/fr/es/... or "auto" (default: de)')
    parser.add_argument('--engine', default=rec.get('engine', 'faster'),
                        choices=['faster', 'openai', 'api'],
                        help='Engine: faster | openai (local) | api (OpenAI-compatible)')
    parser.add_argument('--beam-size', type=int, default=rec.get('beam_size', 5),
                        help='Beam search size: 1=fast, 5=accurate (default: 5)')
    parser.add_argument('--prompt', default=rec.get('prompt', None),
                        help='Context prompt for better recognition of technical terms')
    parser.add_argument('--silence', type=float, default=rec.get('silence', 2.0),
                        help='Silence duration in seconds until auto-stop (default: 2.0)')
    parser.add_argument('--threshold', type=int, default=200,
                        help='Silence RMS threshold (default: 200)')
    parser.add_argument('--api-url', default=None,
                        help='API base URL for engine=api (default: OpenAI)')
    parser.add_argument('--api-key', default=None,
                        help='API key for engine=api (or OPENAI_API_KEY env var)')
    parser.add_argument('--api-model', default=None,
                        help='Model name for engine=api (default: whisper-1)')
    args = parser.parse_args()

    MODEL_SIZE = args.model
    LANGUAGE = None if args.language == "auto" else args.language
    ENGINE = args.engine
    BEAM_SIZE = args.beam_size
    INITIAL_PROMPT = args.prompt
    SILENCE_DURATION = args.silence
    SILENCE_THRESHOLD = args.threshold
    engine_type = args.engine

    _apply_auto_prompt_from_triggers()

    # API configuration (CLI overrides config file)
    if args.api_url:   WHISPER_API_BASE_URL = args.api_url
    if args.api_key:   WHISPER_API_KEY      = args.api_key
    if args.api_model: WHISPER_API_MODEL    = args.api_model

    print()
    print("=" * 60)
    print("🎤 OkaWhisp - System-Level Voice Input")
    print("=" * 60)
    print()
    log.info("=" * 60)
    log.info("🎤 OkaWhisp starting")
    log.info(f"   Log file: {LOG_FILE}")
    log.info("=" * 60)
    log.info(f"  ⚙️  idle_auto_close_seconds={IDLE_CLOSE_SECONDS} | triggers={len(ACTION_TRIGGERS)}")
    log.info(f"  ⚙️  watch: VAD-based | max_seg={WATCH_MAX_SEGMENT_MS}ms silence={WATCH_SILENCE_MS}ms min_seg={WATCH_MIN_SEGMENT_MS}ms")

    # Signal Handler
    def signal_handler(sig, frame):
        global should_exit
        should_exit = True
        print("\n\n👋 OkaWhisp finished.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Control socket (okawhispctl) — starts early, no audio interaction.
    # Watch thread is started after full init (model load + warmup + stream open)
    # to avoid a race where it opens the stream before full init completes.
    _start_control_server()

    # ── Pre-flight checks (before slow model loading) ─────────
    # xdotool is essential for typing — check early, before spending time on model download.
    try:
        subprocess.run(['xdotool', '--version'], capture_output=True, check=True)
        print(f"  ✅ xdotool available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  ❌ xdotool not found! Install via: sudo apt install xdotool")
        sys.exit(1)

    # Wayland warning — xdotool only works on X11
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "wayland":
        print(f"  ⚠️  Wayland detected — xdotool may not work for typing.")
        print(f"     Consider using X11 session or ydotool for Wayland support.")
        log.warning("  Wayland session detected — xdotool typing may fail")
    print()

    # GPU check
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  🖥️  GPU: {gpu_name}")
        else:
            print(f"  ⚠️  No CUDA GPU found, using CPU")
    except ImportError:
        if engine_type == "faster":
            device = "cuda"  # faster-whisper checks itself
        print(f"  ⚠️  PyTorch not available, engine decides device")

    # Check if whisper-server is already running (then no local loading needed)
    _server_available = False
    try:
        import urllib.request
        with urllib.request.urlopen(f"{WHISPER_SERVER_URL}/health", timeout=2) as r:
            if r.read().strip() == b'"ok"':
                _server_available = True
    except Exception:
        pass

    if engine_type == "api":
        # No local model needed — transcription via API
        api_host = WHISPER_API_BASE_URL.split("/")[2] if "//" in WHISPER_API_BASE_URL else WHISPER_API_BASE_URL
        print(f"  🌐 API-Engine: {api_host} (Model: {WHISPER_API_MODEL})")
        if not WHISPER_API_KEY:
            print(f"  ⚠️  No API key set — set OPENAI_API_KEY or use --api-key")
    elif _server_available:
        print(f"  ✅ whisper-server reachable ({WHISPER_SERVER_URL}) — no local model needed")
        print(f"  🔗 Using server GPU for transcription")
    else:
        # Load Whisper model locally
        print(f"  📦 Loading Whisper model '{MODEL_SIZE}' ({engine_type} engine)...")
        if MODEL_SIZE in ["large-v3", "medium"]:
            print(f"     ⏳ First download: ~2-3 GB, this may take a few minutes...")
        print(f"     Progress may not be visible, please be patient.")
        load_start = time.time()
        try:
            if engine_type == "faster":
                model = load_model_faster_whisper(MODEL_SIZE, device)
                print(f"  ✅ faster-whisper loaded ({time.time() - load_start:.1f}s)")
            else:
                model = load_model_openai_whisper(MODEL_SIZE, device)
                print(f"  ✅ openai-whisper loaded ({time.time() - load_start:.1f}s)")
        except Exception as e:
            print(f"  ⚠️  GPU failed ({e}), trying CPU...")
            try:
                if engine_type == "faster":
                    model = load_model_faster_whisper(MODEL_SIZE, "cpu")
                else:
                    model = load_model_openai_whisper(MODEL_SIZE, "cpu")
                print(f"  ✅ Model loaded (CPU Fallback)")
            except Exception as e2:
                print(f"  ❌ Model could not be loaded: {e2}")
                sys.exit(1)

    print()

    # Load silero-vad
    if VAD_ENABLED:
        print(f"  📦 Loading silero-vad model...")
        vad_start = time.time()
        load_vad_model()
        if _vad_model is not None:
            print(f"  ✅ silero-vad loaded ({time.time() - vad_start:.1f}s)")
        else:
            print(f"  ⚠️  silero-vad not available → RMS-Fallback active")
    print()

    # Whisper warmup — first CUDA inference compiles kernels (3–8s).
    # Run a tiny silent dummy through transcribe_faster_whisper now so the
    # first real speech segment is not dropped.
    if engine_type in ("faster", "api") or _server_available:
        print(f"  🔥 Warming up Whisper (first CUDA run)...", end=" ", flush=True)
        try:
            _warmup_audio = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
            transcribe_faster_whisper(_warmup_audio)
            print("done")
        except Exception as _we:
            print(f"skipped ({_we})")
    elif engine_type == "openai" and model is not None:
        print(f"  🔥 Warming up Whisper (first CUDA run)...", end=" ", flush=True)
        try:
            import torch
            _warmup_audio = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
            model.transcribe(_warmup_audio)
            print("done")
        except Exception as _we:
            print(f"skipped ({_we})")
    print()

    vad_status = f"silero-vad (threshold={VAD_THRESHOLD}, silence={VAD_MIN_SILENCE_MS}ms)" \
                 if (VAD_ENABLED and _vad_model is not None) \
                 else f"RMS (threshold={SILENCE_THRESHOLD}, silence={SILENCE_DURATION}s)"
    print()
    print("─" * 60)
    print(f"  ⚙️  Configuration:")
    if engine_type == "api":
        api_host = WHISPER_API_BASE_URL.split("/")[2] if "//" in WHISPER_API_BASE_URL else WHISPER_API_BASE_URL
        print(f"     Engine:      api  ({api_host})")
        print(f"     API-Model:  {WHISPER_API_MODEL}")
    else:
        print(f"     Engine:      {engine_type}-whisper")
        print(f"     Model:      {MODEL_SIZE}")
        print(f"     Beam Size:   {BEAM_SIZE}")
    print(f"     Language:     {LANGUAGE or 'auto-detect'}")
    print(f"     Prompt:      {INITIAL_PROMPT or '(none)'}")
    print(f"     VAD:         {vad_status}")
    print(f"     Max duration: {MAX_RECORD_SECONDS}s")
    print("─" * 60)
    print()

    # parec warm start; close again after idle timeout.
    try:
        _open_audio_stream()
        dev_info = _mic_device_info()
        print(f"  ✅ Audio initialized via parec (idle-close={IDLE_CLOSE_SECONDS}s)")
        log.info(f"  ✅ parec initialized | Input device: {dev_info}")
        _schedule_idle_close(seconds=IDLE_CLOSE_SECONDS, reason="startup-idle")
    except Exception as e:
        log.error(f"  ❌ Audio initialization failed: {e}")
        print(f"  ❌ Audio initialization failed: {e}")
        sys.exit(1)

    # Start hotkey listener
    print("  🔊 Starting hotkey listener...")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Press [{args.key}] to record                │")
    print(f"  │  Speak → Silence → Auto-stop → Type       │")
    print(f"  │  Text is typed into the active window             │")
    print(f"  │                                                 │")
    print(f"  │  Ctrl+C to exit                                 │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    # Signal readiness — plays after model load + warmup, right before hotkey loop.
    play_sound("startup_ready")
    log.info("  ✅ Startup complete — watch mode ready")

    # Start watch thread here, after full init (model loaded, stream opened).
    # This ensures the watch loop gets a clean parec process.
    _start_watch_thread()

    if not listen_keyboard_hotkey(args.key):
        listen_xbindkeys_hotkey(args.key)

    # Cleanup on exit
    _cancel_idle_close_timer()
    _close_audio_stream("shutdown")


if __name__ == "__main__":
    main()
