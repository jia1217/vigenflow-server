#!/usr/bin/env bash
set -euo pipefail

# Disable Git Bash path conversion so MSVC flags aren't mangled
export MSYS_NO_PATHCONV=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting MSVC compilation for vgf-serve..."

# Compiler
CXX="cl"

# Compiler Flags
# Compiler Flags
CXXFLAGS=(
    "/utf-8"
    "/O2"
    "/EHsc"
    "/MD"
    "/std:c++17"
    "/D_WIN32_WINNT=0x0A00"      # Tells Boost to target Windows 10/11
    "/DWIN32_LEAN_AND_MEAN"      # Prevents windows.h from including the legacy WinSock.h
)

# Include Directories (Use forward slashes!)
INCLUDES=(
    "/I" "."
    "/I" "./include"
    "/I" "C:/dev/vcpkg/installed/x64-windows/include"
)

# Source Files
SOURCES=(
    "server_main.cpp"
    "server_config.cpp"
    "file_utils.cpp"
    "request_parser.cpp"
    "z_image_bf16_export.cpp"
    "flux_klein_bf16_export.cpp"
    "worker.cpp"
    "http_handlers.cpp"
    "tcp_server.cpp"
    "cli.cpp"
)

# Linker Flags and Library Paths
LIBPATHS=(
    "/LIBPATH:C:\dev\vcpkg\installed\x64-windows\lib"
)

# Libraries to link against
# Note: ws2_32.lib is strictly required for TCP/networking on Windows
LIBS=(
    "ws2_32.lib"
    "boost_filesystem-vc143-mt-x64-1_88.lib"
)

# Execute the compilation command
# Execute the compilation command
"$CXX" "${CXXFLAGS[@]}" "${INCLUDES[@]}" "${SOURCES[@]}" /Fo"exe_server/" /Fe"exe_server/vgf-serve.exe" /link "${LIBPATHS[@]}" "${LIBS[@]}"

echo "Built $SCRIPT_DIR/exe_server/vgf-serve.exe"