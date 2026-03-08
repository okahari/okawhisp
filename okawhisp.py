#!/usr/bin/env python3
# Requirements: pyaudio, numpy, faster-whisper, silero-vad>=6.0, torch, pynput
# Install via: python3 -m pip install --user pyaudio numpy faster-whisper silero-vad torch pynput
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

import pyaudio
import numpy as np
import subprocess
import threading
import time
import sys
import signal
import argparse
import warnings
import os
import io
import wave
import logging
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

# Console-Handler (nur INFO+)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_log_formatter)
_ch.setLevel(logging.INFO)
log.addHandler(_ch)

# ─── Configuration ───────────────────────────────────────────────

# Whisper requirements (final target values)
SAMPLE_RATE = 16000         # Whisper requires 16kHz
CHUNK_SIZE = 1024           # ~64ms per chunk (at 16kHz)

# Jabra Elite 65t device specifications
DEVICE_SAMPLE_RATE = 48000  # Native Bluetooth microphone sample rate
RESAMPLE_FACTOR = 3         # 48kHz / 16kHz = 3
DEVICE_CHUNK_SIZE = CHUNK_SIZE * RESAMPLE_FACTOR  # = 3072 frames

MODEL_SIZE = "medium"       # medium = best balance for RTX 3060 Ti
LANGUAGE = "de"             # German as default (None = auto-detect)
ENGINE = "faster"           # faster-whisper or openai
BEAM_SIZE = 5               # Beam search for better quality
INITIAL_PROMPT = None       # Context prompt for technical terms
SILENCE_THRESHOLD = 200     # RMS threshold for silence (Jabra BT: speech ~220 RMS)
SILENCE_DURATION = 2.0      # Seconds of silence until auto-stop (RMS fallback)
MIN_RECORD_SECONDS = 1.0    # Minimum recording duration
MAX_RECORD_SECONDS = 120    # Maximum recording duration

# ── silero-vad (Voice Activity Detection) ───────────────────────
# Replaces RMS threshold with ML-based speech recognition.
# No calibration tuning needed — works automatically.
VAD_ENABLED = True          # True = silero-vad, False = RMS fallback
VAD_THRESHOLD = 0.5         # Speech probability threshold (0.0–1.0)
VAD_MIN_SILENCE_MS = 2500   # Silence after last speech until auto-stop
VAD_MIN_SPEECH_MS  = 200    # Minimum speech before stop counter activates
VAD_CHUNK_SAMPLES  = 512    # silero requires exactly 512 samples at 16kHz (32ms)

# Audio ducking: only regulate sink/master volume during recording.
# We intentionally do NOT touch per-app sink-input volumes.
DUCK_AUDIO_DURING_RECORDING = True
DUCK_SINK_LEVEL = 0         # % to reduce sinks to while recording

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

        [duck]
        enabled = true
        sink_level = 10

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

# ─── Globale Variablen ───────────────────────────────────────────

model = None
should_exit = False
engine_type = "faster"
_vad_model = None           # silero-vad Modell (loaded beim Start)

# Statt eines einfachen bool: Lock verhindert Race-Condition bei schnellen Hotkey-Presses
# mehrfach pressed wird (bool-Check + bool-Set ist kein atomarer Vorgang).
_recording_lock = threading.Lock()
_ptt_stop_requested = False  # PTT mode: stop when key released

# Persistente PyAudio-Instanz; Stream wird bei Bedarf geöffnet/geschlossen.
# Startet "warm" und wird nach Idle-Timeout wieder geschlossen (Privacy + GNOME mic indicator).
_pyaudio_instance = None
_audio_stream = None

# Idle close policy (requested): 60s after app start and 60s after each recording stop.
IDLE_CLOSE_SECONDS = 60
_idle_close_timer = None
_idle_timer_lock = threading.Lock()

# ─── ALSA error suppression (once at startup) ──────────────
# WICHTIG: Nur EINMAL setzen, nicht bei jedem Recording!
# Mehrfaches Setzen verursacht SEGV-Crashes
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass  # If ALSA not available


def load_model_faster_whisper(model_size, device):
    """faster-whisper Modell laden (4x schneller)"""
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
    """Original OpenAI Whisper Modell laden"""
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
    """Transkription mit faster-whisper — bevorzugt whisper-server (GPU shared),
    falls back to local instance if server not reachable."""

    # Server-attempt zuerst (Modell readys loaded, GPU)
    result = transcribe_via_server(audio_np)
    if result is not None:
        return result

    # Lokaler Fallback (model muss loaded sein)
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
    """Transkription mit Original OpenAI Whisper"""
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
    """Transcription via OpenAI-kompatibler API.

    Works with:
      - OpenAI (api.openai.com)        → WHISPER_API_MODEL = "whisper-1"
      - Groq   (api.groq.com/openai/v1) → WHISPER_API_MODEL = "whisper-large-v3"
      - lokaler whisper.cpp Server      → WHISPER_API_BASE_URL = "http://localhost:8080"
      - any other OpenAI-compatible endpoint

    Configuration via config file [api] or --api-url / --api-key CLI args.
    API key can also be set as environment variable OPENAI_API_KEY.
    """
    try:
        import urllib.request
        import urllib.error
        import json

        # Audio als WAV in Memory-Buffer schreiben
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
        log.error(f"  [api] ❌ API-Transkription failed: {e}")
        raise


def _play_pcm_sound(samples, sample_rate=44100):
    """Plays a short PCM signal directly via the default output device."""
    if samples is None or len(samples) == 0:
        return False

    try:
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16).tobytes()

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
        )
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return True
    except Exception:
        return False


def _play_audio_file(file_path, volume_boost: float = 1.0, blocking: bool = False):
    """Spielt eine Audio-Datei ab (ffplay bevorzugt, paplay fallback).
    
    Args:
        file_path: Pfad zur Audio-Datei
        volume_boost: Volume multiplier (1.0 = normal, 2.0 = doppelt so laut)
        blocking: If True, wait until playback has finished.
    """
    if not file_path or not os.path.isfile(file_path):
        return False

    # ffplay can play MP3/WAV reliably without GUI.
    try:
        cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']
        if volume_boost != 1.0:
            # Volume filter for volume adjustment
            cmd.extend(['-af', f'volume={volume_boost}'])
        cmd.append(file_path)
        if blocking:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        pass

    # Fallback (works well for WAV/OGA, no volume boost support).
    try:
        if blocking:
            subprocess.run(
                ['paplay', file_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
        else:
            subprocess.Popen(
                ['paplay', file_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        return True
    except FileNotFoundError:
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

    # Zweifach-Puls erzeugt den "Switch"-Charakter statt eines simplen Beeps.
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
    """Leichter elektronischer End-Ton (dezenter Buzzer)."""
    duration = 0.24
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate

    # Sanfter Down-Sweep: signalisiert "Ende" ohne hart zu klingen.
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


def play_sound(sound_name, volume_boost: float = 1.0, blocking: bool = False):
    """Feedback-Sound abspielen.
    
    Args:
        sound_name: Name des Sounds (record_start, record_end)
        volume_boost: Volume multiplier (for sounds during ducking)
        blocking: If True, wait until playback has finished.
    """
    custom_file_map = {
        "record_start": CUSTOM_RECORD_START_SOUND,
        "record_end": CUSTOM_RECORD_END_SOUND,
    }

    # 1) Benutzerdateien haben Vorrang.
    if _play_audio_file(custom_file_map.get(sound_name), volume_boost=volume_boost, blocking=blocking):
        return

    # 2) Synthetic sounds as fallback (no volume boost possible).
    custom_sounds = {
        "record_start": _switch_click_sound,
        "record_end": _soft_end_buzzer_sound,
    }

    builder = custom_sounds.get(sound_name)
    if builder is not None and _play_pcm_sound(builder()):
        return

    # Fallback to system sounds (e.g. if output device temporarily blocked).
    try:
        sound_path = f'/usr/share/sounds/freedesktop/stereo/{sound_name}.oga'
        if blocking:
            subprocess.run(
                ['paplay', sound_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
        else:
            subprocess.Popen(
                ['paplay', sound_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except FileNotFoundError:
        pass


def notify(title, message, urgency="normal"):
    """Desktop-Notification senden (optional)"""
    try:
        subprocess.Popen(
            ['notify-send', '-u', urgency, '-t', '2000', title, message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass


def type_text(text):
    """Text ins activee Fenster tippen via xdotool"""
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
    """Root Mean Square eines Audio-Chunks berechnen"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    if len(audio_data) == 0:
        return 0
    return np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))


DUCK_FADE_STEPS = 15        # Number of steps for smooth volume fade
DUCK_FADE_DURATION = 1.0    # Seconds for fade to 0%
DUCK_PRE_SOUND_PAUSE = 1.0  # Seconds of silence before start sound
DUCK_POST_SOUND_PAUSE = 2.0 # Seconds pause after stop sound before music fades back
SOUND_VOLUME_BOOST = 1.0    # Fixed playback gain for start/stop sounds
SOUND_SINK_LEVEL = 75       # Fixed sink loudness (%) for start/stop sounds


def _parse_sink_inputs_by_pid() -> tuple[list, list]:
    """Parsed 'pactl list sink-inputs', trennt eigene vs. other Streams.

    Returns: (own_inputs, other_inputs) — je Liste von {index, volume_pct, muted}.
    """
    my_pid = str(os.getpid())
    try:
        result = subprocess.run(
            ["pactl", "list", "sink-inputs"],
            capture_output=True, text=True, timeout=3,
            env={**os.environ, "LANG": "C", "LC_ALL": "C"}
        )
    except Exception:
        return [], []

    entries, current = [], {}
    for line in result.stdout.splitlines():
        s = line.strip()
        if s.startswith("Sink Input #"):
            if current.get("index") is not None:
                entries.append(current)
            current = {"index": s.split("#")[1], "volume_pct": 100, "muted": False, "pid": ""}
        elif s.startswith("Mute:"):
            current["muted"] = "yes" in s.lower()
        elif s.startswith("Volume:") and "Base Volume" not in s and "index" in current:
            if "volume_pct" not in current:
                for token in s.split():
                    if token.endswith('%'):
                        try:
                            current["volume_pct"] = int(token.rstrip('%'))
                        except ValueError:
                            pass
                        break
        elif "application.process.id" in s and '"' in s:
            current["pid"] = s.split('"')[1]
    if current.get("index") is not None:
        entries.append(current)

    own    = [e for e in entries if e.get("pid") == my_pid]
    others = [e for e in entries if e.get("pid") != my_pid and not e.get("muted")]
    return own, others


def _get_all_sinks() -> dict:
    """Returns {sink_id: current_volume_pct} for all active sinks."""
    sinks = {}
    try:
        result = subprocess.run(
            ["pactl", "list", "sinks", "short"],
            capture_output=True, text=True, timeout=2
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            sink_id = parts[0]
            vol_result = subprocess.run(
                ["pactl", "get-sink-volume", sink_id],
                capture_output=True, text=True, timeout=2
            )
            for token in vol_result.stdout.split():
                if token.endswith('%'):
                    try:
                        sinks[sink_id] = int(token.rstrip('%'))
                    except ValueError:
                        pass
                    break
    except Exception as e:
        log.warning(f"  [duck] Sink-Liste failed: {e}")
    return sinks


def _fade_sinks_to(sinks: dict, target_pct: int, duration: float = None):
    """Fadet alle Sinks auf einen Ziel-Prozentsatz.
    
    Args:
        sinks: {sink_id: current_volume} - used only for IDs
        target_pct: Target volume in percent (0-100)
        duration: Fade-Dauer in Sekunden (default: DUCK_FADE_DURATION)
    """
    if not sinks:
        return
    if duration is None:
        duration = DUCK_FADE_DURATION
    
    try:
        # Get current volumes for smooth fade
        current_vols = _get_all_sinks()
        step_delay = duration / DUCK_FADE_STEPS
        
        for step in range(1, DUCK_FADE_STEPS + 1):
            factor = step / DUCK_FADE_STEPS
            for sid in sinks:
                current = current_vols.get(sid, 100)
                vol = int(current + (target_pct - current) * factor)
                subprocess.run(["pactl", "set-sink-volume", sid, f"{vol}%"],
                               capture_output=True, timeout=2)
            time.sleep(step_delay)
        
        # Finale Werte setzen
        for sid in sinks:
            subprocess.run(["pactl", "set-sink-volume", sid, f"{target_pct}%"],
                           capture_output=True, timeout=2)
        log.debug(f"  [fade] Sinks auf {target_pct}% gefadet ({duration:.1f}s)")
    except Exception as e:
        log.warning(f"  [fade] Fade failed: {e}")


def _set_sink_volumes(sinks: dict, volumes: dict):
    """Setzt Sink-Volumes direkt (ohne Fade).
    
    Args:
        sinks: {sink_id: ...} - used only for IDs
        volumes: {sink_id: volume_pct} - Ziel-Volumes
    """
    for sid in sinks:
        vol = volumes.get(sid, 100)
        try:
            subprocess.run(["pactl", "set-sink-volume", sid, f"{vol}%"],
                           capture_output=True, timeout=2)
        except Exception as e:
            log.warning(f"  [set-vol] Error for sink {sid}: {e}")


def _get_other_sink_inputs() -> dict:
    """Gibt {input_index: volume_pct} for all other sink inputs.
    
    Filters out our own process (ffplay for sounds).
    """
    my_pid = str(os.getpid())
    inputs = {}
    
    try:
        # LANG=C for English output (independent of system locale)
        result = subprocess.run(
            ["pactl", "list", "sink-inputs"],
            capture_output=True, text=True, timeout=3,
            env={**os.environ, "LANG": "C", "LC_ALL": "C"}
        )
        
        current = {}
        for line in result.stdout.splitlines():
            s = line.strip()
            if s.startswith("Sink Input #"):
                if current.get("index") is not None and current.get("pid") != my_pid:
                    inputs[current["index"]] = current.get("volume_pct", 100)
                current = {"index": s.split("#")[1], "volume_pct": 100, "pid": ""}
            elif s.startswith("Volume:") and "Base Volume" not in s:
                for token in s.split():
                    if token.endswith('%'):
                        try:
                            current["volume_pct"] = int(token.rstrip('%'))
                        except ValueError:
                            pass
                        break
            elif "application.process.id" in s and '"' in s:
                current["pid"] = s.split('"')[1]
        
        # Letzten Eintrag
        if current.get("index") is not None and current.get("pid") != my_pid:
            inputs[current["index"]] = current.get("volume_pct", 100)
            
    except Exception as e:
        log.warning(f"  [duck] Sink-Input-Liste failed: {e}")
    
    return inputs


def _fade_sink_inputs_to(inputs: dict, target_pct: int, duration: float = None):
    """Fadet alle sink inputs auf einen Ziel-Prozentsatz.
    
    Args:
        inputs: {input_index: original_volume} 
        target_pct: Target volume in percent (0-100)
        duration: Fade-Dauer in Sekunden
    """
    if not inputs:
        return
    if duration is None:
        duration = DUCK_FADE_DURATION
    
    try:
        step_delay = duration / DUCK_FADE_STEPS
        
        for step in range(1, DUCK_FADE_STEPS + 1):
            factor = step / DUCK_FADE_STEPS
            for idx, orig in inputs.items():
                vol = int(orig + (target_pct - orig) * factor)
                subprocess.run(["pactl", "set-sink-input-volume", idx, f"{vol}%"],
                               capture_output=True, timeout=2)
            time.sleep(step_delay)
        
        # Finale Werte setzen
        for idx in inputs:
            subprocess.run(["pactl", "set-sink-input-volume", idx, f"{target_pct}%"],
                           capture_output=True, timeout=2)
        log.debug(f"  [fade] {len(inputs)} sink inputs auf {target_pct}% gefadet ({duration:.1f}s)")
    except Exception as e:
        log.warning(f"  [fade] Sink-Input-Fade failed: {e}")


def _restore_sink_inputs(inputs: dict, duration: float = None):
    """Restores sink inputs to their original volume (with fade)."""
    if not inputs:
        return
    if duration is None:
        duration = DUCK_FADE_DURATION
    
    try:
        # Aktuelle Volumes holen (LANG=C for English output)
        current = {}
        result = subprocess.run(
            ["pactl", "list", "sink-inputs"],
            capture_output=True, text=True, timeout=3,
            env={**os.environ, "LANG": "C", "LC_ALL": "C"}
        )
        cur_entry = {}
        for line in result.stdout.splitlines():
            s = line.strip()
            if s.startswith("Sink Input #"):
                if cur_entry.get("index"):
                    current[cur_entry["index"]] = cur_entry.get("volume_pct", 0)
                cur_entry = {"index": s.split("#")[1], "volume_pct": 0}
            elif s.startswith("Volume:") and "Base Volume" not in s:
                for token in s.split():
                    if token.endswith('%'):
                        try:
                            cur_entry["volume_pct"] = int(token.rstrip('%'))
                        except ValueError:
                            pass
                        break
        if cur_entry.get("index"):
            current[cur_entry["index"]] = cur_entry.get("volume_pct", 0)
        
        step_delay = duration / DUCK_FADE_STEPS
        
        for step in range(1, DUCK_FADE_STEPS + 1):
            factor = step / DUCK_FADE_STEPS
            for idx, orig in inputs.items():
                cur = current.get(idx, 0)
                vol = int(cur + (orig - cur) * factor)
                subprocess.run(["pactl", "set-sink-input-volume", idx, f"{vol}%"],
                               capture_output=True, timeout=2)
            time.sleep(step_delay)
        
        # Finale Original-Werte setzen
        for idx, orig in inputs.items():
            subprocess.run(["pactl", "set-sink-input-volume", idx, f"{orig}%"],
                           capture_output=True, timeout=2)
        log.debug(f"  [restore] {len(inputs)} sink inputs restored ({duration:.1f}s)")
    except Exception as e:
        log.warning(f"  [restore] Sink-Input-Restore failed: {e}")


def calibrate_silence_threshold(audio_stream, duration_s: float = 0.8) -> int:
    """Misst den aktuellen Noise-Floor und leitet einen dynamischen Threshold ab.

    Liest `duration_s` Sekunden Audio (~12 Chunks bei 0.8s) und verwendet das
    20. percentile of RMS values — more robust than median, since early speech
    or short outliers (single loud chunks) do not skew the value.

    Minimum: 40 (for very quiet environments).
    Maximum: 400 (Deckel als Sicherheitsnetz).
    """
    n_chunks = max(1, int(duration_s * SAMPLE_RATE / CHUNK_SIZE))
    rms_values = []
    for _ in range(n_chunks):
        try:
            data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms_values.append(get_rms(data))
        except Exception:
            pass
    if not rms_values:
        return SILENCE_THRESHOLD
    # 20. Perzentil: nimmt die leisesten 20% der Chunks → echter Silence-Floor
    # Median would be too high due to early speech.
    # Factor 4 (instead of 8): keeps threshold well below normal speech level (~150–400 RMS)
    noise_floor = float(np.percentile(rms_values, 20))
    dynamic = int(noise_floor * 4)
    dynamic = max(40, min(dynamic, 250))
    log.info(f"  [2b] Kalibrierung: noise_floor(p20)={noise_floor:.1f} → threshold={dynamic} "
             f"(min={min(rms_values):.1f}, max={max(rms_values):.1f}, n={len(rms_values)})")
    return dynamic


def record_with_silence_detection(audio_stream, threshold: int = None):
    """Nimmt Audio auf bis Silence erkannt wird"""
    effective_threshold = threshold if threshold is not None else SILENCE_THRESHOLD
    frames = []
    silent_chunks = 0
    # Berechnungen mit Whisper Sample-Rate (16kHz - Hardware macht Resampling)
    chunks_for_silence = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    min_chunks = int(MIN_RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
    # Warmup: Silence-Counter starting erst nach 1.5s.
    # Gibt dem Device Zeit zum Stabilisieren UND stellt sicher dass der
    # Silence timer never runs before user hears the start sound.
    warmup_chunks = int(1.5 * SAMPLE_RATE / CHUNK_SIZE)
    total_chunks = 0

    while total_chunks < max_chunks:
        try:
            # Lese 16kHz chunks (Hardware-Resampling von 48kHz)
            data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
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
            print(f"  Audio-Error: {e}")
            break

    return frames


def record_with_vad(audio_stream, ptt_mode: bool = False) -> list:
    """Nimmt Audio auf mit silero-vad Voice Activity Detection.

    Liest 512-Sample Chunks (32ms bei 16kHz), analysiert jeden per silero-vad.
    Stoppt wenn nach erkannter Sprache VAD_MIN_SILENCE_MS Silence folgt.
    Returns all recorded bytes (incl. pre-speech frames) for Whisper.
    
    ptt_mode: If True, stop when _ptt_stop_requested becomes True (key released).

    State-Machine:
        WAITING  → warte auf erste Sprache (aber akkumuliere Audio)
        SPEAKING → Speech detected, silence counter running
        STOPPED  → Silence-Timeout erreicht (or PTT key released)
    """
    import torch
    global _ptt_stop_requested

    vad_chunk = VAD_CHUNK_SAMPLES                            # 512 Samples = 32ms
    ms_per_chunk = vad_chunk / SAMPLE_RATE * 1000           # = 32ms
    silence_chunks_needed = int(VAD_MIN_SILENCE_MS / ms_per_chunk)
    min_speech_chunks     = int(VAD_MIN_SPEECH_MS  / ms_per_chunk)
    min_total_chunks      = int(MIN_RECORD_SECONDS * 1000 / ms_per_chunk)
    max_total_chunks      = int(MAX_RECORD_SECONDS * 1000 / ms_per_chunk)

    frames          = []
    speech_chunks   = 0   # konsekutive Sprach-Chunks
    silence_streak  = 0   # konsekutive Silence-Chunks nach Sprachbeginn
    speech_started  = False
    total_chunks    = 0

    while total_chunks < max_total_chunks:
        # PTT mode: stop immediately when key released
        if ptt_mode and _ptt_stop_requested:
            elapsed_ms = total_chunks * ms_per_chunk
            log.info(f"  PTT: 🛑 Key released → Stopp (t={elapsed_ms:.0f}ms)")
            break
        try:
            data = audio_stream.read(vad_chunk, exception_on_overflow=False)
        except Exception as e:
            log.warning(f"  VAD read-Error: {e}")
            break

        frames.append(data)
        total_chunks += 1

        # Float32 tensor [512] for silero-vad
        audio_f32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        with torch.no_grad():
            speech_prob = _vad_model(torch.from_numpy(audio_f32), SAMPLE_RATE).item()

        is_speech = speech_prob >= VAD_THRESHOLD

        if is_speech:
            speech_chunks += 1
            silence_streak = 0
            if not speech_started and speech_chunks >= min_speech_chunks:
                speech_started = True
                elapsed_ms = total_chunks * ms_per_chunk
                log.debug(f"  VAD: 🗣  Speech detected (prob={speech_prob:.2f}, t={elapsed_ms:.0f}ms)")
        else:
            speech_chunks = 0
            if speech_started:
                silence_streak += 1
                if (total_chunks >= min_total_chunks and
                        silence_streak >= silence_chunks_needed):
                    elapsed_ms = total_chunks * ms_per_chunk
                    log.debug(f"  VAD: 🤫 Silence {silence_streak * ms_per_chunk:.0f}ms → Stopp "
                              f"(t={elapsed_ms:.0f}ms)")
                    break

    return frames


def _mic_device_info() -> str:
    """Returns the name of the default input device (for logging)."""
    try:
        info = _pyaudio_instance.get_default_input_device_info()
        return f"{info.get('name', '?')} (idx={info.get('index', '?')})"
    except Exception as e:
        return f"<unbekannt: {e}>"


def _check_mic_ready(max_wait_s: float = 3.0) -> bool:
    """Checks if the stream delivers real audio data (kein all-zeros Buffer).

    After Bluetooth reconnect or PipeWire suspend the device delivers
    kurze Zeit nur Nullen. Diese Funktion wartet bis real data ankommen
    or max_wait_s is exceeded.

    Returns True wenn ready, False bei Timeout.
    """
    deadline = time.time() + max_wait_s
    check_chunks = int(0.1 * SAMPLE_RATE / CHUNK_SIZE)  # ~0.1s pro Check-Runde
    attempts = 0
    consecutive_ok = 0  # Anzahl aufeinanderfolgender checks mit RMS > 50

    while time.time() < deadline:
        attempts += 1
        has_signal = False
        max_rms = 0.0

        for _ in range(check_chunks):
            try:
                data = _audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                rms = get_rms(data)
                if rms > max_rms:
                    max_rms = rms
            except Exception as ex:
                log.warning(f"  MIC-CHECK read-Error (attempt {attempts}): {ex}")
                break

        # Stufen:
        #   ZEROS       RMS < 2   → Device komplett inactive
        #   TRANSITIONING RMS 2-8 → Jabra wechselt Profil (A2DP→SCO), aber nutzbar
        #   OK          RMS > 8   → stable signal, ready for recording
        # Experience: RMS 5-7 during transition → recording works anyway (RMS 600+)
        if max_rms > 8:
            consecutive_ok += 1
        else:
            consecutive_ok = 0

        status = "OK" if max_rms > 8 else ("TRANSITIONING" if max_rms > 2 else "ZEROS")
        log.debug(f"  MIC-CHECK attempt {attempts}: max_rms={max_rms:.1f} → {status} (consecutive_ok={consecutive_ok})")

        if consecutive_ok >= 1:
            log.info(f"  ✅ Microphone ready nach {attempts} Check(s), RMS={max_rms:.1f}")
            return True

        wait = min(0.2, deadline - time.time())
        if wait > 0:
            time.sleep(wait)

    log.warning(f"  ⚠️  Microphone NICHT ready nach {max_wait_s:.1f}s ({attempts} checks) — starting anyway")
    return False


def _open_audio_stream():
    """(Re)opens input stream when needed."""
    global _pyaudio_instance, _audio_stream

    if _pyaudio_instance is None:
        _pyaudio_instance = pyaudio.PyAudio()

    if _audio_stream is None:
        _audio_stream = _pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        log.info(f"  ✅ Input stream opened | Device: {_mic_device_info()}")


def _close_audio_stream(reason: str = "idle-timeout"):
    """Closes input stream fully (removes active mic handle in PipeWire)."""
    global _audio_stream

    if _audio_stream is None:
        return

    try:
        if _audio_stream.is_active():
            _audio_stream.stop_stream()
    except Exception:
        pass

    try:
        _audio_stream.close()
    except Exception as ex:
        log.warning(f"  ⚠️  close_stream failed ({reason}): {ex}")
    finally:
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
        if _recording_lock.locked():
            _schedule_idle_close(seconds=seconds, reason=reason)
            return
        _close_audio_stream(reason=reason)

    with _idle_timer_lock:
        _idle_close_timer = threading.Timer(seconds, _idle_close_task)
        _idle_close_timer.daemon = True
        _idle_close_timer.start()
    log.info(f"  ⏳ Idle close scheduled in {seconds}s ({reason})")


def do_voice_input(ptt_mode: bool = False, skip_start_sound: bool = False):
    """Hauptfunktion: Aufnehmen → Transcribingn → Typing
    
    ptt_mode: If True, Push-to-Talk mode (stop on key release, send with Enter)
    skip_start_sound: If True, start sound was already played (PTT mode)
    """
    global _ptt_stop_requested

    # Lock statt bool: verhindert Race-Condition bei schnell-mehrfachem Hotkey
    if not _recording_lock.acquire(blocking=False):
        log.debug("Hotkey ignored — Recording already running (Lock gehalten)")
        return

    # Reset PTT stop flag
    _ptt_stop_requested = False
    
    mode_str = "PTT" if ptt_mode else "Toggle"
    log.info("━" * 60)
    log.info(f"▶  Recording sequence starting ({mode_str} mode)")
    try:
        _run_voice_input(ptt_mode=ptt_mode, skip_start_sound=skip_start_sound)
    finally:
        _ptt_stop_requested = False
        _recording_lock.release()
        _schedule_idle_close(seconds=IDLE_CLOSE_SECONDS, reason="post-record-idle")
        log.info("◀  Recording sequence finished")
        log.info("━" * 60)


def _run_voice_input(ptt_mode: bool = False, skip_start_sound: bool = False):
    """Actual recording logic (runs under _recording_lock).
    
    ptt_mode: If True, PTT mode - stop on key release, send with Enter at end.
    skip_start_sound: If True, start sound was already played externally.
    """
    global _audio_stream

    _cancel_idle_close_timer()
    _open_audio_stream()

    # ── 1. Stream-Status checkingn & starten ──────────────────────
    stream_active = _audio_stream.is_active() if _audio_stream else False
    stream_stopped = _audio_stream.is_stopped() if _audio_stream else True
    log.info(f"  [1] Stream status before start: active={stream_active}, stopped={stream_stopped}")
    log.info(f"  [1] Input-Device: {_mic_device_info()}")

    try:
        _audio_stream.start_stream()
        log.info(f"  [1] start_stream() OK → active={_audio_stream.is_active()}")
    except Exception as e:
        log.error(f"  [1] ❌ start_stream() FEHLER: {e}")
        notify("❌ Audio-Error", str(e), "critical")
        return

    # ── 2. Device-Readiness checkingn (Bluetooth-Reconnect abwarten) ──
    log.info(f"  [2] Checking ob Microphone real data delivers...")
    mic_ready = _check_mic_ready(max_wait_s=3.0)
    log.info(f"  [2] Mic-Ready: {mic_ready} | Device: {_mic_device_info()}")

    if not mic_ready:
        log.warning("  [2] Mic not ready — versuche Stream to reopen...")
        try:
            _audio_stream.stop_stream()
            _audio_stream.close()
            _audio_stream = _pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
            log.info("  [2] Stream neu opened — second Readiness-Check...")
            mic_ready = _check_mic_ready(max_wait_s=2.0)
            log.info(f"  [2] Zweiter check: mic_ready={mic_ready}")
        except Exception as e:
            log.error(f"  [2] ❌ Stream-Neustart failed: {e}")
            notify("❌ Audio-Error", f"Mic not ready: {e}", "critical")
            return

    # ── 2b. Sink-Volumes erfassen (kein per-app Ducking) ─────────
    _original_sinks = {}
    if DUCK_AUDIO_DURING_RECORDING:
        _original_sinks = _get_all_sinks()
        log.info(f"  [2b] 📊 Sinks captured: {len(_original_sinks)}")

    # ── 2c. Sink-Volumes absenken (sanfter Übergang) ─────────────
    # Skip in PTT mode (too slow, sound already played)
    if not skip_start_sound:
        if DUCK_AUDIO_DURING_RECORDING and _original_sinks:
            log.info(f"  [2c] 🔉 Fade: Sinks → {DUCK_SINK_LEVEL}% ({DUCK_FADE_DURATION}s)")
            _fade_sinks_to(_original_sinks, DUCK_SINK_LEVEL)
            # Silence vor dem Start-Sound
            log.info(f"  [2c] 🤫 Silence pause ({DUCK_PRE_SOUND_PAUSE}s)")
            time.sleep(DUCK_PRE_SOUND_PAUSE)

        # ── 2d. Start-Sound
        log.info(f"  [2d] 🔔 Start-Sound @ {SOUND_SINK_LEVEL}%")
        _chime_other_inputs = {}
        if DUCK_AUDIO_DURING_RECORDING and _original_sinks:
            # Keep background apps muted while temporarily raising sink for chime.
            _chime_other_inputs = _get_other_sink_inputs()
            if _chime_other_inputs:
                _fade_sink_inputs_to(_chime_other_inputs, 0, duration=0.05)
            _set_sink_volumes(_original_sinks, {sid: SOUND_SINK_LEVEL for sid in _original_sinks})
        play_sound("record_start")
        notify("🔴 Recording", "Speak now...")
        time.sleep(0.5)   # Sound fertig abspielen lassen
        if DUCK_AUDIO_DURING_RECORDING and _original_sinks:
            _set_sink_volumes(_original_sinks, {sid: DUCK_SINK_LEVEL for sid in _original_sinks})
            if _chime_other_inputs:
                _restore_sink_inputs(_chime_other_inputs, duration=0.05)
    else:
        log.info("  [2d] ⏭️  Start-Sound skipped (PTT mode)")
        notify("🔴 PTT Recording", "Speak...")

    # ── 2e. Kalibrierung (nur RMS-Fallback) ──────────────────────
    use_vad = VAD_ENABLED and _vad_model is not None
    if not use_vad:
        dynamic_threshold = calibrate_silence_threshold(_audio_stream)
        log.info(f"  [2e] RMS-Fallback: threshold={dynamic_threshold}")
    else:
        dynamic_threshold = None

    if use_vad:
        log.info(f"  [3] 🔴 Recording (VAD, threshold={VAD_THRESHOLD}, "
                 f"min_silence={VAD_MIN_SILENCE_MS}ms, sink_level={DUCK_SINK_LEVEL}%)")
    else:
        log.info(f"  [3] 🔴 Recording (RMS, Silence-Limit={SILENCE_DURATION}s, "
                 f"Threshold={dynamic_threshold}, sink_level={DUCK_SINK_LEVEL}%)")

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
        try:
            _audio_stream.stop_stream()
            log.info(f"  [4] ⏹  Stream stopped nach {elapsed_record:.2f}s | active={_audio_stream.is_active()}")
        except Exception as ex:
            log.warning(f"  [4] stop_stream() Error: {ex}")

    # ── 5. Stop-Sound (Music ist noch geduckt) ─────────────────────
    # In PTT mode, stop sound was already played on key release
    if not ptt_mode:
        log.info(f"  [5] 🔔 Stop-Sound @ {SOUND_SINK_LEVEL}%")
        _chime_other_inputs = {}
        if DUCK_AUDIO_DURING_RECORDING and _original_sinks:
            _chime_other_inputs = _get_other_sink_inputs()
            if _chime_other_inputs:
                _fade_sink_inputs_to(_chime_other_inputs, 0, duration=0.05)
            _set_sink_volumes(_original_sinks, {sid: SOUND_SINK_LEVEL for sid in _original_sinks})
        play_sound("record_end")
        time.sleep(0.5)
        if DUCK_AUDIO_DURING_RECORDING and _original_sinks:
            _set_sink_volumes(_original_sinks, {sid: DUCK_SINK_LEVEL for sid in _original_sinks})
            if _chime_other_inputs:
                _restore_sink_inputs(_chime_other_inputs, duration=0.05)
    else:
        log.info("  [5] ⏭️  Stop-Sound skipped (PTT mode, already played)")

    # ── 5b. Pause after stop sound, dann Sinks auf Originalwerte ───
    if DUCK_AUDIO_DURING_RECORDING and _original_sinks:
        log.info(f"  [5b] ⏸️  Pause after stop sound ({DUCK_POST_SOUND_PAUSE}s)")
        time.sleep(DUCK_POST_SOUND_PAUSE)
        log.info(f"  [5c] 🔊 Restoring sink volumes ({DUCK_FADE_DURATION}s)")
        _set_sink_volumes(_original_sinks, _original_sinks)
    else:
        time.sleep(0.3)

    if not frames:
        log.warning("  [5] No audio frames recorded")
        notify("⚠️ No recording", "No audio data")
        return

    # ── 6. Audio auswerten ───────────────────────────────────────
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio_np) / SAMPLE_RATE
    rms_overall = float(np.sqrt(np.mean(audio_np ** 2))) * 32768
    log.info(f"  [6] Audio: {duration:.2f}s, {len(frames)} Chunks, RMS_gesamt={rms_overall:.1f}")

    if duration < 0.5:
        log.warning("  [6] Recording too short (<0.5s) — discarded")
        return

    # ── 7. Transkription ─────────────────────────────────────────
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
        log.error(f"  [7] ❌ Transkriptions-Error: {e}")
        notify("❌ Error", str(e), "critical")
        return

    if not text:
        log.warning(f"  [7] No speech detected (Dauer={duration:.1f}s, RMS={rms_overall:.1f})")
        notify("⚠️ No text", "No speech detected")
        return

    log.info(f"  [7] 📝 [{lang}] ({elapsed_trans:.1f}s) {text}")

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
    """Horcht auf globalen Hotkey via pynput"""
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
            108,    # Some systems report raw keycode 108 for AltGr
        }
        is_altgr_hotkey = hotkey_norm in ('ALT_GR', 'ALTGR', 'RIGHT_ALT', 'RALT')
        
        if not target_key and not is_altgr_hotkey:
            print(f"  ❌ Unbekannter Hotkey: {hotkey}")
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
                        # Play start sound IMMEDIATELY
                        play_sound("record_start", volume_boost=SOUND_VOLUME_BOOST)
                        thread = threading.Thread(target=lambda: do_voice_input(ptt_mode=True, skip_start_sound=True), daemon=True)
                        thread.start()
                
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
                # PTT mode: stop recording (handled in do_voice_input via flag)
                log.info(f"◀  PTT release after {press_duration:.1f}s — Stopping")
                # Play stop sound IMMEDIATELY
                play_sound("record_end", volume_boost=SOUND_VOLUME_BOOST)
                global _ptt_stop_requested
                _ptt_stop_requested = True
            elif press_duration < ptt_state['ptt_threshold'] and not _recording_lock.locked():
                # Short press: toggle mode (original behavior)
                log.info("▶  Short press — Toggle mode")
                thread = threading.Thread(target=lambda: do_voice_input(ptt_mode=False), daemon=True)
                thread.start()

        print(f"  🎹 Hotkey: {hotkey}")
        print(f"  📍 Methode: pynput (global keyboard listener)")
        print(f"  📍 PTT-Modus: Taste ≥300ms halten = Push-to-Talk")

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    except ImportError:
        return False

    return True


def listen_xbindkeys_hotkey(hotkey):
    """Fallback: Enter-basiert"""
    print(f"  ⚠️  pynput nicht installiert.")
    print(f"  Installiere: pip install pynput")
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
    global DUCK_AUDIO_DURING_RECORDING, DUCK_SINK_LEVEL
    global CUSTOM_RECORD_START_SOUND, CUSTOM_RECORD_END_SOUND
    global WHISPER_API_BASE_URL, WHISPER_API_KEY, WHISPER_API_MODEL

    # ── Config File laden (Defaults) ─────────────────────────────
    cfg = load_config()
    rec = cfg.get("recording", {})
    vad = cfg.get("vad", {})
    duck = cfg.get("duck", {})
    sounds = cfg.get("sounds", {})
    api_cfg = cfg.get("api", {})

    # Config values as defaults (CLI overrides these)
    if vad.get("enabled") is not None:    VAD_ENABLED        = vad["enabled"]
    if vad.get("threshold"):              VAD_THRESHOLD      = float(vad["threshold"])
    if vad.get("min_silence_ms"):         VAD_MIN_SILENCE_MS = int(vad["min_silence_ms"])
    if duck.get("enabled") is not None:   DUCK_AUDIO_DURING_RECORDING = duck["enabled"]
    if duck.get("sink_level") is not None: DUCK_SINK_LEVEL   = int(duck["sink_level"])
    if sounds.get("start"):               CUSTOM_RECORD_START_SOUND = sounds["start"]
    if sounds.get("stop"):                CUSTOM_RECORD_END_SOUND   = sounds["stop"]
    if api_cfg.get("base_url"):           WHISPER_API_BASE_URL = api_cfg["base_url"]
    if api_cfg.get("api_key"):            WHISPER_API_KEY      = api_cfg["api_key"]
    if api_cfg.get("model"):              WHISPER_API_MODEL    = api_cfg["model"]

    parser = argparse.ArgumentParser(
        description='OkaWhisp - System-Level Voice Input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python okawhisp.py                                       # Standard (AltGr, medium, deutsch)
  python okawhisp.py --model small --language en           # Englisch, schneller
  python okawhisp.py --engine api --api-key sk-...         # OpenAI API
  python okawhisp.py --engine api \\
    --api-url https://api.groq.com/openai/v1 --api-key gsk_  # Groq (kostenlos, schnell)
  python voice-type.py --prompt "NestJS, Flutter, API"       # Fachbegriffe als Kontext
  python voice-type.py --beam-size 1                         # Schneller, etwas ungenauer

Config File: ~/.config/voice-type/config.toml
        """
    )
    parser.add_argument('--key', default=rec.get('key', 'ALT_GR'),
                        help='Recording hotkey (ALT_GR or F1-F12, default: ALT_GR)')
    parser.add_argument('--model', default=rec.get('model', 'medium'),
                        help='Whisper Model: tiny/base/small/medium/large-v3 (default: medium)')
    parser.add_argument('--language', default=rec.get('language', 'de'),
                        help='Language: de/en/fr/es/... oder "auto" (default: de)')
    parser.add_argument('--engine', default=rec.get('engine', 'faster'),
                        choices=['faster', 'openai', 'api'],
                        help='Engine: faster | openai (lokal) | api (OpenAI-kompatibel)')
    parser.add_argument('--beam-size', type=int, default=rec.get('beam_size', 5),
                        help='Beam search size: 1=fast, 5=accurate (default: 5)')
    parser.add_argument('--prompt', default=rec.get('prompt', None),
                        help='Context prompt for better recognition of technical terms')
    parser.add_argument('--silence', type=float, default=rec.get('silence', 2.0),
                        help='Silence-Dauer in Sekunden bis Auto-stop (default: 2.0)')
    parser.add_argument('--threshold', type=int, default=200,
                        help='Silence-Schwellwert RMS (default: 200)')
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
    log.info("🎤 OkaWhisp gestarting")
    log.info(f"   Log-Datei: {LOG_FILE}")
    log.info("=" * 60)

    # Signal Handler
    def signal_handler(sig, frame):
        global should_exit
        should_exit = True
        print("\n\n👋 OkaWhisp finished.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # GPU checkingn
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  🖥️  GPU: {gpu_name}")
        else:
            print(f"  ⚠️  Keine CUDA GPU, nutze CPU")
    except ImportError:
        if engine_type == "faster":
            device = "cuda"  # faster-whisper checks itself
        print(f"  ⚠️  PyTorch not available, Engine entscheidet Device")

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
            print(f"  ⚠️  Kein API-Key gesetzt — setze OPENAI_API_KEY oder --api-key")
    elif _server_available:
        print(f"  ✅ whisper-server reachable ({WHISPER_SERVER_URL}) — no local model needed")
        print(f"  🔗 Using server GPU for transcription")
    else:
        # Whisper-Modell lokal laden
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
            print(f"  ⚠️  GPU failed ({e}), versuche CPU...")
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

    # silero-vad laden
    if VAD_ENABLED:
        print(f"  📦 Loading silero-vad model...")
        vad_start = time.time()
        load_vad_model()
        if _vad_model is not None:
            print(f"  ✅ silero-vad loaded ({time.time() - vad_start:.1f}s)")
        else:
            print(f"  ⚠️  silero-vad not available → RMS-Fallback active")
    print()

    # xdotool checkingn
    try:
        subprocess.run(['xdotool', '--version'], capture_output=True, check=True)
        print(f"  ✅ xdotool available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  ❌ xdotool nicht found! sudo apt install xdotool")
        sys.exit(1)

    vad_status = f"silero-vad (threshold={VAD_THRESHOLD}, silence={VAD_MIN_SILENCE_MS}ms)" \
                 if (VAD_ENABLED and _vad_model is not None) \
                 else f"RMS (threshold={SILENCE_THRESHOLD}, silence={SILENCE_DURATION}s)"
    print()
    print("─" * 60)
    print(f"  ⚙️  Konfiguration:")
    if engine_type == "api":
        api_host = WHISPER_API_BASE_URL.split("/")[2] if "//" in WHISPER_API_BASE_URL else WHISPER_API_BASE_URL
        print(f"     Engine:      api  ({api_host})")
        print(f"     API-Model:  {WHISPER_API_MODEL}")
    else:
        print(f"     Engine:      {engine_type}-whisper")
        print(f"     Model:      {MODEL_SIZE}")
        print(f"     Beam Size:   {BEAM_SIZE}")
    print(f"     Language:     {LANGUAGE or 'auto-detect'}")
    print(f"     Prompt:      {INITIAL_PROMPT or '(keiner)'}")
    print(f"     VAD:         {vad_status}")
    print(f"     Max Dauer:   {MAX_RECORD_SECONDS}s")
    print("─" * 60)
    print()

    # PyAudio-Instanz + Stream warm starten; nach 60s idle wieder schließen.
    global _pyaudio_instance, _audio_stream
    try:
        _open_audio_stream()
        _audio_stream.stop_stream()  # warm start, no active capture
        dev_info = _mic_device_info()
        print(f"  ✅ PyAudio + Input-Stream initialisiert (warm start, idle-close={IDLE_CLOSE_SECONDS}s)")
        log.info(f"  ✅ PyAudio initialisiert | Input-Device: {dev_info}")
        _schedule_idle_close(seconds=IDLE_CLOSE_SECONDS, reason="startup-idle")
    except Exception as e:
        log.error(f"  ❌ Audio-Initialisierung failed: {e}")
        print(f"  ❌ Audio-Initialisierung failed: {e}")
        sys.exit(1)

    # Hotkey-Listener starten
    print("  🔊 Starte Hotkey-Listener...")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Press [{args.key}] to record                │")
    print(f"  │  Speak → Silence → Auto-stop → Type       │")
    print(f"  │  Text wird ins activee Fenster getippt           │")
    print(f"  │                                                 │")
    print(f"  │  Ctrl+C zum Beenden                             │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    if not listen_keyboard_hotkey(args.key):
        listen_xbindkeys_hotkey(args.key)

    # Cleanup beim Beenden
    _cancel_idle_close_timer()
    if _audio_stream is not None:
        try:
            _audio_stream.close()
        except:
            pass
    if _pyaudio_instance is not None:
        try:
            _pyaudio_instance.terminate()
        except:
            pass


if __name__ == "__main__":
    main()
