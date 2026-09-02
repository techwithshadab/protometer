#!/bin/sh
# Start as root only to fix the ownership of mounted volumes, then drop to the unprivileged
# `botox` user for the actual process. Named volumes (the HuggingFace cache, the data dir) mount
# owned by root the first time, which the non-root app cannot write; chown them here so the
# embedding-model download and the index build succeed. Idempotent and cheap.
set -e

CACHE_DIR="${HF_HOME:-/home/botox/.cache/huggingface}"
mkdir -p "$CACHE_DIR" /app/data/raw /app/data/processed /app/data/index

# Only chown when running as root (compose default). If the image is ever run with --user, skip.
if [ "$(id -u)" = "0" ]; then
  chown -R botox:botox "$CACHE_DIR" /app/data 2>/dev/null || true
  # setpriv drops the uid to `botox` but does NOT change $HOME, which is inherited as /root from this
  # root-started entrypoint. Several libraries (the Protegrity appython SDK among them) look for
  # config under $HOME/.protegrity; with HOME=/root the non-root botox user gets a PermissionError
  # statting root's home and protection fails to init. Export the correct HOME before dropping priv.
  export HOME=/home/botox
  exec setpriv --reuid botox --regid botox --init-groups env "HOME=/home/botox" "$@"
fi

# Already unprivileged (e.g. run with --user botox): just exec.
exec "$@"
