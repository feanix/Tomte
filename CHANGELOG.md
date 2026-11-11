# Changelog

## 1.1.0

- New: parts with their own texture sets. Move each part of the source
  mesh into its own UDIM tile (head in 1001, teeth in 1002 ...) and every
  output is put together from one texture per tile, with a fill colour for
  parts that have no texture. In the app the texture list gets a column per
  tile; on the command line, UDIM-numbered files are grouped by themselves
  and `--map` covers other names. Setup files store these outputs too.
- Each tile's normal map format is detected on its own, and converted to
  match the largest part when they differ.

## 1.0.1

- Fixed: the desktop app could crash on Python 3.10 and 3.11 with PySide6
  6.12.0, because of a bug in that PySide6 release. On those Python versions
  the requirements now install an earlier PySide6, and the app explains how
  to fix an install that already has 6.12.0.

## 1.0.0 (first public release)

- Desktop app (`Tomte.bat` or `python -m uvtransfer.gui`): pick two FBX files,
  drop in any number of textures, preview both layouts with UV overlays.
- Setup files (`.tomte`): save meshes, textures, types and settings; reopen
  them in the app or rerun them with `tomte_cli.py --setup`. The app can
  copy a ready-made command line.
- Reload (F5) re-reads meshes and textures and marks results whose
  textures changed.
- Command line (`tomte_cli.py`, or `python -m uvtransfer`) and Python API.
- Colour textures are resampled in linear light; data textures as-is.
- Tangent-space normal maps are re-oriented for rotated, mirrored and
  rescaled islands. DirectX/OpenGL format is detected automatically.
- ASCII and binary FBX (6.x and 7.x), several meshes per file, any UV channel.
- 8- and 16-bit PNG, TIFF, JPEG, TGA and more; 16-bit output.
- Warnings for overlapping islands, UVs far outside 0 to 1 and meshes that
  don't match; a memory check before large transfers.
- Damaged or hostile FBX files are rejected with a clear message instead of
  hanging or exhausting memory.
- Sample sphere in `samples/` to try it on.
