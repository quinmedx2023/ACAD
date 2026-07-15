# OCCT Fusion Cleaner

C++ sidecar for cleaning Autodesk Fusion 360 Gallery Segmentation STEP files into
the ACAD dataset layout.

Build:

Using the official Windows OCCT package extracted as `C:\tools\OpenCascade`:

```powershell
cmake -S tools/occt_cleaner -B tools/occt_cleaner/build `
  -DCMAKE_BUILD_TYPE=Release `
  -DACAD_OCCT_ROOT=C:\tools\OpenCascade
cmake --build tools/occt_cleaner/build
```

`ACAD_OCCT_ROOT` should point at the OCCT package root containing `inc`,
`cmake`, and `win64\vc14\lib`. If `OpenCASCADE_DIR` is set to
`C:\tools\OpenCascade\cmake`, CMake will infer this root automatically.

Using an installed OCCT CMake package:

```powershell
cmake -S tools/occt_cleaner -B tools/occt_cleaner/build `
  -DOpenCASCADE_DIR=<path-to-occt-cmake-config>
cmake --build tools/occt_cleaner/build --config Release
```

For the official Windows prebuilt package, prefer `ACAD_OCCT_ROOT`; its
`OpenCASCADEConfig.cmake` may import Visualization/Draw targets that require
VTK even though this cleaner only needs STEP/BRep libraries.

Or let CMake download and build OCCT 8.0.0:

```powershell
cmake -S tools/occt_cleaner -B tools/occt_cleaner/build `
  -DACAD_FETCH_OCCT=ON
cmake --build tools/occt_cleaner/build --config Release
```

The fetch path downloads:

```text
https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V8_0_0.zip
```

Run through Rust:

```powershell
cargo run -p acad-brep-candle-train -- clean-fusion `
  --raw raw/Fusion360GalleryDataset/segmentation `
  --out data/fusion-seg-v1 `
  --limit 100 `
  --allow-boundary
```

The Rust wrapper automatically adds OCCT runtime DLL directories to the sidecar
process `PATH` when it can detect `ACAD_OCCT_ROOT`, `OpenCASCADE_DIR`, `CASROOT`,
or the default `C:\tools\OpenCascade` layout. It also sets the key OCCT resource
environment variables used by STEP import, such as `CASROOT`, `CSF_XSMessage`,
and `CSF_STEPDefaults`. For this local package, the third-party DLL root is:

```text
C:\tools\OpenCascade\3rdparty
```

Run directly:

```powershell
tools/occt_cleaner/build/occt_cleaner.exe `
  --raw raw/Fusion360GalleryDataset/segmentation `
  --out data/fusion-seg-v1 `
  --limit 100 `
  --allow-boundary
```

If running directly fails to find OCCT DLLs, initialize the official OCCT
runtime environment or add the same DLL directories to `PATH`. Because this
workspace uses `C:\tools\OpenCascade\3rdparty`, set `THIRDPARTY_DIR` before
calling `env.bat`:

```powershell
cmd /c "set THIRDPARTY_DIR=C:\tools\OpenCascade\3rdparty&& call C:\tools\OpenCascade\env.bat vc143 64&& tools\occt_cleaner\build\occt_cleaner.exe --help"
```

If you build with a Visual Studio multi-config generator instead of Ninja, the
executable will usually be under `tools\occt_cleaner\build\Release`.
