# Unified VigenFlow Server

This folder keeps one server source tree for both Ubuntu and Windows. The runtime code is shared; only the compile command and worker executable name are platform-specific.

## Ubuntu

```bash
bash build_ubuntu.sh
./vgf-serve -p 1128
```

Ubuntu builds `./vgf-serve` with `g++`. The model worker binaries are expected to be named `run.exe` under `exe_models/...`.

## Windows

```bat
build_windows.bat
exe_server\vgf-serve.exe -p 1128
```

Windows builds `exe_server\vgf-serve.exe` with MSVC. The model worker binaries are expected to be named `host.exe` under `exe_models\...`.

The script also copies the matching Boost filesystem/system DLLs from vcpkg into `exe_server` when they are available.

If vcpkg is not installed at `C:\dev\vcpkg`, set `VCPKG_ROOT` before running:

```bat
set VCPKG_ROOT=D:\path\to\vcpkg
build_windows.bat
```

If your Boost filesystem library has a different filename, set `BOOST_FILESYSTEM_LIB`:

```bat
set BOOST_FILESYSTEM_LIB=boost_filesystem-vc143-mt-x64-1_89.lib
build_windows.bat
```

## Auto Dispatcher

```bash
bash build_vgf_serve.sh
```

The dispatcher calls `build_windows.sh` on Windows/Git Bash and `build_ubuntu.sh` on Ubuntu.

## CI/CD

The repo-level workflow `.github/workflows/ci-cd.yml` builds this folder on Ubuntu and Windows, runs `vgf-serve --help`, uploads artifacts, and creates release packages when a version tag such as `v1.0.0` is pushed.
