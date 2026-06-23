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

GitHub Actions is configured in `.github/workflows/ci-cd.yml`.

It runs on every push, pull request, and manual workflow dispatch:

- Builds `server_unified` on Ubuntu with `build_ubuntu.sh`
- Builds `server_unified` on Windows with `build_windows.bat`
- Runs `vgf-serve --help` as a smoke test
- Uploads `vgf-serve-ubuntu-x64` and `vgf-serve-windows-x64` build artifacts

To create a GitHub Release with packaged binaries, push a tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The release job creates:

- `vgf-serve-v1.0.0-ubuntu-x64.tar.gz`
- `vgf-serve-v1.0.0-windows-x64.zip`
