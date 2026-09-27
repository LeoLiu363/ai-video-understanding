#!/usr/bin/env bash
# 在云服务器上安装 / 更新 vedioAI
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/vedioai}"
PY="${PY:-python3}"

echo "==> 目标目录: $APP_DIR"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "安装 Python3…"
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y python3 python3-venv python3-pip ffmpeg git
  elif command -v yum >/dev/null 2>&1; then
    yum install -y python3 python3-pip ffmpeg git
  else
    echo "请先手动安装 python3 / ffmpeg / git" >&2
    exit 1
  fi
fi

# 确保 ffmpeg 存在（yt-dlp / 代理需要）
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y ffmpeg
  fi
fi

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[ocr]"

mkdir -p data
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo "已从 .env.example 生成 .env，请填入 API Key："
    echo "  nano $APP_DIR/.env"
  else
    touch .env
  fi
fi

echo "==> 安装完成"
echo "启动："
echo "  cd $APP_DIR && source .venv/bin/activate && vedioai serve --host 0.0.0.0 --port 17831"
echo "安全组请放行 TCP 17831"
