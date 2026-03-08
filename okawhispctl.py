#!/usr/bin/env python3
import argparse
import json
import socket
import sys
from pathlib import Path

SOCKET_PATH = str(Path.home() / ".local" / "share" / "okawhisp" / "control.sock")


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


def main():
    parser = argparse.ArgumentParser(prog="okawhispctl")
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

    args = parser.parse_args()

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
