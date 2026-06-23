#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

g++ -std=c++17 \
  server_main.cpp \
  server_config.cpp \
  file_utils.cpp \
  request_parser.cpp \
  z_image_bf16_export.cpp \
  flux_klein_bf16_export.cpp \
  worker.cpp \
  http_handlers.cpp \
  tcp_server.cpp \
  cli.cpp \
  -I. -I./include \
  -pthread \
  -lboost_system \
  -o vgf-serve

echo "Built $SCRIPT_DIR/vgf-serve"
