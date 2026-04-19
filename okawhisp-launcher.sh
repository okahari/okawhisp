#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "okawhisp-launcher: python3 not found" >&2
  exit 1
fi

INSTALL_DIR="${OKAWHISP_INSTALL_DIR:-$HOME/.local/share/okawhisp}"
SCRIPT_PATH="${INSTALL_DIR}/okawhisp.py"
USER_ID="$(id -u)"
USER_NAME="${USER:-$(id -un)}"

session_id="${XDG_SESSION_ID:-}"
if [[ -z "${session_id}" ]]; then
  session_id="$(loginctl list-sessions --no-legend 2>/dev/null | awk -v user="${USER_NAME}" '$3==user {print $1; exit}')"
fi

display_name=""
if [[ -n "${session_id}" ]]; then
  display_name="$(loginctl show-session "${session_id}" -p Display --value 2>/dev/null | tr -d '\n')"
fi
if [[ -z "${display_name}" ]]; then
  if compgen -G "/tmp/.X11-unix/X*" >/dev/null 2>&1; then
    latest_socket="$(ls -1 /tmp/.X11-unix/X* 2>/dev/null | sort | tail -n1)"
    if [[ -n "${latest_socket}" ]]; then
      display_name=":${latest_socket##*/X}"
    fi
  fi
fi
if [[ -z "${display_name}" ]]; then
  display_name=":0"
fi

remote_flag=""
if [[ -n "${session_id}" ]]; then
  remote_flag="$(loginctl show-session "${session_id}" -p Remote --value 2>/dev/null | tr -d '\n')"
fi

if [[ -n "${XAUTHORITY:-}" ]]; then
  xauthority_path="${XAUTHORITY}"
else
  if [[ "${remote_flag}" == "yes" ]]; then
    xauthority_path="/run/user/${USER_ID}/gdm/Xauthority"
  elif [[ -f "${HOME}/.Xauthority" ]]; then
    xauthority_path="${HOME}/.Xauthority"
  else
    xauthority_path="/run/user/${USER_ID}/gdm/Xauthority"
  fi
fi

export DISPLAY="${display_name}"
export XAUTHORITY="${xauthority_path}"
export XDG_RUNTIME_DIR="/run/user/${USER_ID}"
export PYTHONUNBUFFERED=1

echo "okawhisp-launcher: session=${session_id:-unknown} display=${display_name} xauth=${xauthority_path} remote=${remote_flag:-unknown}" >&2

exec "${PYTHON_BIN}" "${SCRIPT_PATH}"
