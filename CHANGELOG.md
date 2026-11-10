# Changelog

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
