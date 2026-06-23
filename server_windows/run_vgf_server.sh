#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Edit these fixed options when you want to run a different model/path.
# MODEL_NAME is the worker target. OpenWebUI will show/select LoRA ids as ${MODEL_NAME}-<id>.
MODEL_NAME="flux.2-klein-4B-lora"
DISPLAY_MODEL_NAME="flux.2-klein-4B-lora"
WEIGHTS_PATH=""
NPU_FILES_PATH=""
PORT="11281"
EXE_BASE_DIR=""
EXE_PATH=""
# Set EXE_PATH only when you want to force every request through one executable.
# EXE_PATH="/home/kelsey/NPU_projects/NPU_repo/deb_lib/vigenflow_1.4_amd64/opt/vigenflow/exe_models/z-image-turbo/lora_add_q41/run.exe"

# LoRA options.
# Set either LORA_SOURCE to a Hugging Face repo/file URL, or LORA_DIR to an existing bin folder.
# Add Q41 Hugging Face LoRAs to LORA_MODEL_LIST_Q41.
# Add BF16 Hugging Face LoRAs to LORA_MODEL_LIST_BF16. The base model still
# appears as z-image-turbo-bf16; only per-LoRA choices use bf16-lora IDs.
# Add FLUX LoRAs to LORA_MODEL_LIST_FLUX or LORA_MODEL_LIST_FLUX_EDIT.
# Set LORA_CATALOG_DIR to a parent/export folder for downloaded or existing LoRA bins.
# LORA_SOURCE="ABDALLALSWAITI/Spectra-Etch"
LORA_SOURCE=""
# LORA_SOURCE=""
LORA_DIR=""
LORA_FILE=""
# LORA_REVISION="main"
LORA_CATALOG_DIR=""
LORA_MODEL_LIST_Q41=""
LORA_MODEL_LIST_BF16=""
LORA_MODEL_LIST_FLUX=""
LORA_MODEL_LIST_FLUX_EDIT=""
LORA_MODEL_LIST=""
# LORA_MODEL_LIST="$SCRIPT_DIR/custom_lora_models.json"
# LORA_CACHE_DIR="/home/kelsey/Igpu/Z-Image/src/loras"
LORA_RANK=32

# GGUF options for z-image-turbo-Q4_1-GGUF and z-image-turbo-Q4_1-lora only.
# Leave these empty to use the server defaults:
#   unsloth/Z-Image-Turbo-GGUF / z-image-turbo-Q4_1.gguf
# Set GGUF_PATH to pack a local GGUF instead.
GGUF_PATH=""
GGUF_REPO=""
GGUF_FILE=""
GGUF_REVISION="main"
GGUF_HF_TOKEN=""
FORCE_GGUF_DOWNLOAD="false"

# BF16/shared-weight options for Z-Image BF16 and Q4_1 first-run bootstrap.
# Download/export is implemented in C++.
Z_IMAGE_BF16_REPO="Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_BF16_REVISION="main"
Z_IMAGE_BF16_CACHE_DIR=""
Z_IMAGE_BF16_HF_TOKEN=""
FORCE_Z_IMAGE_BF16_DOWNLOAD="false"
NO_Z_IMAGE_BF16_AUTO_DOWNLOAD="false"

# BF16 options for Flux.2-klein-4B. Prefer Kelsey1217/FLUX.2-klein-4B-npu;
# missing denoising/text files fall back to Black Forest raw safetensors.
FLUX_KLEIN_BF16_REPO="Kelsey1217/FLUX.2-klein-4B-npu"
FLUX_KLEIN_BF16_REVISION="main"
FLUX_KLEIN_BF16_CACHE_DIR=""
FLUX_KLEIN_BF16_HF_TOKEN=""
FORCE_FLUX_KLEIN_BF16_DOWNLOAD="false"
NO_FLUX_KLEIN_BF16_AUTO_DOWNLOAD="false"

GGUF_ARGS=()
[[ -n "$GGUF_PATH" ]] && GGUF_ARGS+=(--gguf-path "$GGUF_PATH")
[[ -n "$GGUF_REPO" ]] && GGUF_ARGS+=(--gguf-repo "$GGUF_REPO")
[[ -n "$GGUF_FILE" ]] && GGUF_ARGS+=(--gguf-file "$GGUF_FILE")
[[ -n "$GGUF_REVISION" && "$GGUF_REVISION" != "main" ]] && GGUF_ARGS+=(--gguf-revision "$GGUF_REVISION")
[[ -n "$GGUF_HF_TOKEN" ]] && GGUF_ARGS+=(--gguf-hf-token "$GGUF_HF_TOKEN")
case "$FORCE_GGUF_DOWNLOAD" in
  1|true|TRUE|yes|YES|on|ON) GGUF_ARGS+=(--force-gguf-download) ;;
esac

Z_IMAGE_BF16_ARGS=()
[[ -n "$Z_IMAGE_BF16_REPO" ]] && Z_IMAGE_BF16_ARGS+=(--z-image-bf16-repo "$Z_IMAGE_BF16_REPO")
[[ -n "$Z_IMAGE_BF16_REVISION" && "$Z_IMAGE_BF16_REVISION" != "main" ]] && Z_IMAGE_BF16_ARGS+=(--z-image-bf16-revision "$Z_IMAGE_BF16_REVISION")
[[ -n "$Z_IMAGE_BF16_CACHE_DIR" ]] && Z_IMAGE_BF16_ARGS+=(--z-image-bf16-cache-dir "$Z_IMAGE_BF16_CACHE_DIR")
[[ -n "$Z_IMAGE_BF16_HF_TOKEN" ]] && Z_IMAGE_BF16_ARGS+=(--z-image-bf16-hf-token "$Z_IMAGE_BF16_HF_TOKEN")
case "$FORCE_Z_IMAGE_BF16_DOWNLOAD" in
  1|true|TRUE|yes|YES|on|ON) Z_IMAGE_BF16_ARGS+=(--force-z-image-bf16-download) ;;
esac
case "$NO_Z_IMAGE_BF16_AUTO_DOWNLOAD" in
  1|true|TRUE|yes|YES|on|ON) Z_IMAGE_BF16_ARGS+=(--no-z-image-bf16-auto-download) ;;
esac

FLUX_KLEIN_BF16_ARGS=()
[[ -n "$FLUX_KLEIN_BF16_REPO" ]] && FLUX_KLEIN_BF16_ARGS+=(--flux-klein-bf16-repo "$FLUX_KLEIN_BF16_REPO")
[[ -n "$FLUX_KLEIN_BF16_REVISION" && "$FLUX_KLEIN_BF16_REVISION" != "main" ]] && FLUX_KLEIN_BF16_ARGS+=(--flux-klein-bf16-revision "$FLUX_KLEIN_BF16_REVISION")
[[ -n "$FLUX_KLEIN_BF16_CACHE_DIR" ]] && FLUX_KLEIN_BF16_ARGS+=(--flux-klein-bf16-cache-dir "$FLUX_KLEIN_BF16_CACHE_DIR")
[[ -n "$FLUX_KLEIN_BF16_HF_TOKEN" ]] && FLUX_KLEIN_BF16_ARGS+=(--flux-klein-bf16-hf-token "$FLUX_KLEIN_BF16_HF_TOKEN")
case "$FORCE_FLUX_KLEIN_BF16_DOWNLOAD" in
  1|true|TRUE|yes|YES|on|ON) FLUX_KLEIN_BF16_ARGS+=(--force-flux-klein-bf16-download) ;;
esac
case "$NO_FLUX_KLEIN_BF16_AUTO_DOWNLOAD" in
  1|true|TRUE|yes|YES|on|ON) FLUX_KLEIN_BF16_ARGS+=(--no-flux-klein-bf16-auto-download) ;;
esac

for list_path in "$LORA_MODEL_LIST_Q41" "$LORA_MODEL_LIST_BF16" "$LORA_MODEL_LIST_FLUX" "$LORA_MODEL_LIST_FLUX_EDIT" "$LORA_MODEL_LIST"; do
  if [[ -n "$list_path" && ! -f "$list_path" ]]; then
    echo "Missing LoRA model list: $list_path" >&2
    exit 1
  fi
done

LORA_ARGS=()
[[ -n "$LORA_SOURCE" ]] && LORA_ARGS+=(--lora-source "$LORA_SOURCE")
[[ -n "$LORA_RANK" ]] && LORA_ARGS+=(--lora-rank "$LORA_RANK")
# [[ -n "$LORA_REVISION" ]] && LORA_ARGS+=(--lora-revision "$LORA_REVISION")
[[ -n "$LORA_DIR" ]] && LORA_ARGS+=(--lora-dir "$LORA_DIR")
[[ -n "$LORA_CATALOG_DIR" ]] && LORA_ARGS+=(--lora-catalog-dir "$LORA_CATALOG_DIR")
[[ -n "$LORA_MODEL_LIST_Q41" && -f "$LORA_MODEL_LIST_Q41" ]] && LORA_ARGS+=(--lora-model-list-q41 "$LORA_MODEL_LIST_Q41")
[[ -n "$LORA_MODEL_LIST_BF16" && -f "$LORA_MODEL_LIST_BF16" ]] && LORA_ARGS+=(--lora-model-list-bf16 "$LORA_MODEL_LIST_BF16")
[[ -n "$LORA_MODEL_LIST_FLUX" && -f "$LORA_MODEL_LIST_FLUX" ]] && LORA_ARGS+=(--lora-model-list-flux "$LORA_MODEL_LIST_FLUX")
[[ -n "$LORA_MODEL_LIST_FLUX_EDIT" && -f "$LORA_MODEL_LIST_FLUX_EDIT" ]] && LORA_ARGS+=(--lora-model-list-flux-edit "$LORA_MODEL_LIST_FLUX_EDIT")
[[ -n "$LORA_MODEL_LIST" && -f "$LORA_MODEL_LIST" ]] && LORA_ARGS+=(--lora-model-list "$LORA_MODEL_LIST")
[[ -n "$LORA_FILE" ]] && LORA_ARGS+=(--lora-file "$LORA_FILE")
# [[ -n "$LORA_CACHE_DIR" ]] && LORA_ARGS+=(--lora-cache-dir "$LORA_CACHE_DIR")
# [[ -n "$LORA_EXPORT_DIR" ]] && LORA_ARGS+=(--lora-export-dir "$LORA_EXPORT_DIR")

# case "$LORA_FORCE" in
#   1|true|TRUE|yes|YES|on|ON) LORA_ARGS+=(--lora-force) ;;
# esac

# case "$LORA_KEEP_CACHE" in
#   1|true|TRUE|yes|YES|on|ON) LORA_ARGS+=(--lora-keep-cache) ;;
# esac

VGF_SERVE_EXE="$SCRIPT_DIR/exe_server/vgf-serve.exe"
if [[ ! -f "$VGF_SERVE_EXE" ]]; then
  VGF_SERVE_EXE="$SCRIPT_DIR/vgf-serve.exe"
fi
if [[ ! -f "$VGF_SERVE_EXE" ]]; then
  echo "Missing vgf-serve executable. Build it first with build_vgf_serve.sh." >&2
  exit 1
fi

VGF_ARGS=("$MODEL_NAME" -p "$PORT")
[[ -n "$WEIGHTS_PATH" ]] && VGF_ARGS+=(-w "$WEIGHTS_PATH")
[[ -n "$NPU_FILES_PATH" ]] && VGF_ARGS+=(-n "$NPU_FILES_PATH")
[[ -n "$EXE_BASE_DIR" ]] && VGF_ARGS+=(--exe-base-dir "$EXE_BASE_DIR")
[[ -n "$EXE_PATH" ]] && VGF_ARGS+=(-e "$EXE_PATH")
[[ -n "$DISPLAY_MODEL_NAME" ]] && VGF_ARGS+=(-m "$DISPLAY_MODEL_NAME")

exec "$VGF_SERVE_EXE" "${VGF_ARGS[@]}" "${GGUF_ARGS[@]}" "${Z_IMAGE_BF16_ARGS[@]}" "${FLUX_KLEIN_BF16_ARGS[@]}" "${LORA_ARGS[@]}" "$@"
