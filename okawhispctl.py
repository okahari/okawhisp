#!/usr/bin/env python3
import argparse
import importlib.util
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

SOCKET_PATH = str(Path.home() / ".local" / "share" / "okawhisp" / "control.sock")
SERVICE_PATH = Path.home() / ".config" / "systemd" / "user" / "okawhisp.service"
CONFIG_PATH = Path.home() / ".config" / "okawhisp" / "config.toml"


def send(req: dict):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(SOCKET_PATH)
        s.sendall(json.dumps(req).encode("utf-8"))
        data = s.recv(65536)
        if not data:
            return {"ok": False, "message": "empty response"}
        return json.loads(data.decode("utf-8", errors="ignore"))
    finally:
        s.close()


def _read_config():
    """Read config.toml and return as dict (requires tomllib or toml)."""
    if not CONFIG_PATH.exists():
        return {}
    txt = CONFIG_PATH.read_text()
    try:
        import tomllib
        return tomllib.loads(txt)
    except ImportError:
        pass
    try:
        import tomli
        return tomli.loads(txt)
    except ImportError:
        pass
    # Minimal fallback: parse key = "value" lines
    cfg = {}
    for line in txt.splitlines():
        m = re.match(r'^\s*(\w+)\s*=\s*"([^"]*)"', line)
        if m:
            cfg[m.group(1)] = m.group(2)
    return cfg


def _write_config_value(section: str, key: str, value: str):
    """Update a single key in config.toml, preserving structure."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(f"[{section}]\n{key} = \"{value}\"\n")
        return
    txt = CONFIG_PATH.read_text()
    pattern = rf'^(\s*{re.escape(key)}\s*=\s*)"[^"]*"'
    new_txt, count = re.subn(pattern, rf'\1"{value}"', txt, flags=re.MULTILINE)
    if count == 0:
        # Key not found — append to section or create section
        if f"[{section}]" in txt:
            new_txt = txt.replace(f"[{section}]", f"[{section}]\n{key} = \"{value}\"")
        else:
            new_txt = txt.rstrip() + f"\n\n[{section}]\n{key} = \"{value}\"\n"
    CONFIG_PATH.write_text(new_txt)


def _current_engine_model():
    cfg = _read_config()
    rec = cfg.get("recording", cfg)  # flat fallback
    return rec.get("engine"), rec.get("model")


def _cache_present(engine: str, model: str):
    if not engine or not model:
        return False
    if engine == "openai":
        return (Path.home() / ".cache" / "whisper" / f"{model}.pt").exists()
    if engine == "faster":
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        prefix = f"models--Systran--faster-whisper-{model}"
        return any(p.name.startswith(prefix) for p in hub.glob("models--Systran--faster-whisper-*") if p.is_dir())
    return False


def _engine_available(engine: str):
    if engine == "openai":
        return importlib.util.find_spec("whisper") is not None
    if engine == "faster":
        return importlib.util.find_spec("faster_whisper") is not None
    return False


def model_status():
    engine, model = _current_engine_model()
    engines = {
        "faster": {
            "available": _engine_available("faster"),
            "active": engine == "faster",
            "cachePresent": _cache_present("faster", model or "") if engine == "faster" else None,
        },
        "openai": {
            "available": _engine_available("openai"),
            "active": engine == "openai",
            "cachePresent": _cache_present("openai", model or "") if engine == "openai" else None,
        },
    }
    return {
        "ok": True,
        "activeEngine": engine,
        "activeModel": model,
        "serviceFile": str(SERVICE_PATH),
        "engines": engines,
    }


def _print_model_status(resp: dict):
    print("OkaWisp Model Status")
    print(f"- Active: {resp.get('activeEngine')} / {resp.get('activeModel')}")
    print(f"- Service file: {resp.get('serviceFile')}")
    print("- Engines:")
    engines = resp.get("engines", {})
    for name in ("faster", "openai"):
        e = engines.get(name, {})
        avail = "yes" if e.get("available") else "no"
        active = "(active)" if e.get("active") else ""
        cache = e.get("cachePresent")
        cache_s = "n/a" if cache is None else ("yes" if cache else "no")
        print(f"  - {name}: available={avail}, cache={cache_s} {active}".rstrip())


def model_pull(engine: str, model: str):
    if engine == "openai":
        code = (
            "import whisper; "
            f"print('Pulling openai-whisper model: {model}'); "
            f"whisper.load_model('{model}', device='cpu'); "
            "print('Done')"
        )
    elif engine == "faster":
        code = (
            "from faster_whisper import WhisperModel; "
            f"print('Pulling faster-whisper model: {model}'); "
            f"WhisperModel('{model}', device='cpu', compute_type='int8'); "
            "print('Done')"
        )
    else:
        return {"ok": False, "message": f"unsupported engine: {engine}"}

    proc = subprocess.run(["python3", "-c", code], text=True)
    return {"ok": proc.returncode == 0, "engine": engine, "model": model, "exit": proc.returncode}


def model_set(engine: str, model: str):
    _write_config_value("recording", "engine", engine)
    _write_config_value("recording", "model", model)

    subprocess.run(["systemctl", "--user", "reset-failed", "okawhisp.service"], check=False)
    restart = subprocess.run(["systemctl", "--user", "restart", "okawhisp.service"], check=False)

    return {
        "ok": restart.returncode == 0,
        "engine": engine,
        "model": model,
        "message": "config updated, service restarted" if restart.returncode == 0 else "config updated, restart failed",
    }


def main():
    parser = argparse.ArgumentParser(prog="okawhisp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    watch = sub.add_parser("watch")
    watch_sub = watch.add_subparsers(dest="watch_cmd", required=True)
    watch_sub.add_parser("start")
    watch_sub.add_parser("stop")

    dur = watch_sub.add_parser("duration")
    dur.add_argument("value", help="e.g. 5m, 60s, 1h")

    wset = watch_sub.add_parser("set")
    wset.add_argument("--idle-close", required=True, dest="idle_close", help="e.g. 60s")

    test = sub.add_parser("test")
    test_sub = test.add_subparsers(dest="test_cmd", required=True)
    trg = test_sub.add_parser("trigger")
    trg.add_argument("name")
    trg.add_argument("--text", required=True)

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="model_cmd", required=True)
    ms_status = model_sub.add_parser("status")
    ms_status.add_argument("--json", action="store_true", dest="as_json")
    mp = model_sub.add_parser("pull")
    mp.add_argument("engine", choices=["faster", "openai"])
    mp.add_argument("model")
    ms = model_sub.add_parser("set")
    ms.add_argument("engine", choices=["faster", "openai"])
    ms.add_argument("model")

    args = parser.parse_args()

    # local model management commands (no daemon socket required)
    if args.cmd == "model":
        if args.model_cmd == "status":
            resp = model_status()
            if getattr(args, "as_json", False):
                print(json.dumps(resp, ensure_ascii=False))
            else:
                _print_model_status(resp)
            if not resp.get("ok", False):
                sys.exit(1)
            return
        elif args.model_cmd == "pull":
            resp = model_pull(args.engine, args.model)
        elif args.model_cmd == "set":
            resp = model_set(args.engine, args.model)
        else:
            parser.error("unknown model command")
        print(json.dumps(resp, ensure_ascii=False))
        if not resp.get("ok", False):
            sys.exit(1)
        return

    # daemon commands
    if args.cmd == "status":
        req = {"op": "status"}
    elif args.cmd == "watch":
        if args.watch_cmd == "start":
            req = {"op": "watch.start"}
        elif args.watch_cmd == "stop":
            req = {"op": "watch.stop"}
        elif args.watch_cmd == "duration":
            req = {"op": "watch.duration", "duration": args.value}
        elif args.watch_cmd == "set":
            req = {"op": "watch.set_idle_close", "idle_close": args.idle_close}
        else:
            parser.error("unknown watch command")
    elif args.cmd == "test":
        if args.test_cmd == "trigger":
            req = {"op": "test.trigger", "name": args.name, "text": args.text}
        else:
            parser.error("unknown test command")
    else:
        parser.error("unknown command")

    try:
        resp = send(req)
    except Exception as ex:
        print(json.dumps({"ok": False, "message": str(ex)}))
        sys.exit(1)

    print(json.dumps(resp, ensure_ascii=False))
    if not resp.get("ok", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
